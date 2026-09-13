#!/usr/bin/env python3
"""
Live Nifty tape sidecar — prices + VWAP + EFI + CVD + VA triggers only.

Run next to your Streamlit app (separate process):

    pip install smartapi-python pyotp websocket-client pandas numpy python-dotenv
    python live_tape_server.py

Open http://127.0.0.1:8765
Does not compute GEX / VEX / IV / heatmap.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
import urllib.request
from collections import deque
from datetime import date
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Laptop: 127.0.0.1:8765. Render: HOST=0.0.0.0 and PORT is set by Render.
HOST = os.getenv("TAPE_HOST", "0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
PORT = int(os.getenv("PORT") or os.getenv("TAPE_PORT", "8765"))
BAR_SECONDS = int(os.getenv("TAPE_BAR_SECONDS", "180"))  # 3-min bars by default
NIFTY_TOKEN = os.getenv("NIFTY_TOKEN", "99926000")
# Optional override. If empty, nearest Nifty futures token is downloaded from Angel.
FUT_TOKEN = os.getenv("FUT_TOKEN", "")
FUT_EXCHANGE_TYPE = int(os.getenv("FUT_EXCHANGE_TYPE", "2"))  # NFO = 2
SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"


def resolve_nifty_fut_token() -> str:
    """Nearest unexpired NIFTY index future on NFO. Same idea as the Streamlit app."""
    if FUT_TOKEN:
        return str(FUT_TOKEN)
    try:
        with urllib.request.urlopen(SCRIP_MASTER_URL, timeout=30) as resp:
            rows = json.loads(resp.read().decode())
        today = date.today()
        best = None
        for r in rows:
            if str(r.get("name", "")).upper() != "NIFTY":
                continue
            if str(r.get("exch_seg", "")).upper() != "NFO":
                continue
            if str(r.get("instrumenttype", "")).upper() != "FUTIDX":
                continue
            exp = pd.to_datetime(r.get("expiry"), format="%d%b%Y", errors="coerce")
            if pd.isna(exp) or exp.date() < today:
                continue
            tok = str(r.get("token") or "")
            if not tok:
                continue
            if best is None or exp.date() < best[0]:
                best = (exp.date(), tok, r.get("symbol"))
        if best:
            print(f"auto futures token {best[2]} expiry {best[0]} token {best[1]}")
            return best[1]
    except Exception as e:
        print("fut token lookup failed", e)
    return ""

HERE = Path(__file__).resolve().parent
HTML_PATH = HERE / "live_tape.html"

# ---------- shared snapshot ----------
LOCK = threading.Lock()
SNAPSHOT = {
    "spot": None,
    "fut": None,
    "vwap": None,
    "efi": None,
    "cvd": None,
    "bar_tf": f"{BAR_SECONDS // 60} min",
    "bars": [],
    "va": {"action": "NO ENTRY", "micro": "starting"},
}
CLIENTS: set[asyncio.Queue] = set()
LOOP: asyncio.AbstractEventLoop | None = None


def _publish():
    if LOOP is None:
        return
    payload = json.dumps(SNAPSHOT, default=str)
    for q in list(CLIENTS):
        try:
            LOOP.call_soon_threadsafe(q.put_nowait, payload)
        except Exception:
            pass


# ---------- incremental session bars ----------
class SessionBars:
    def __init__(self, bar_seconds: int):
        self.bar_seconds = bar_seconds
        self.cur = None
        self.rows = []
        self.pv = 0.0
        self.vol = 0.0
        self.cvd = 0.0
        self.prev_close = None

    def _floor(self, ts: float) -> int:
        return int(ts) - (int(ts) % self.bar_seconds)

    def on_tick(self, ts: float, price: float, volume: float = 0.0, spot: float | None = None):
        if price <= 0:
            return
        t0 = self._floor(ts)
        signed = 0.0
        if self.prev_close is not None:
            signed = volume * (1.0 if price >= self.prev_close else -1.0)
        if self.cur is None or t0 != self.cur["t0"]:
            if self.cur is not None:
                self.rows.append(dict(self.cur))
                if len(self.rows) > 240:
                    self.rows = self.rows[-240:]
            self.cur = {
                "t0": t0,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "spot": spot if spot else price,
                "volume": 0.0,
                "delta": 0.0,
            }
        self.cur["high"] = max(self.cur["high"], price)
        self.cur["low"] = min(self.cur["low"], price)
        self.cur["close"] = price
        if spot:
            self.cur["spot"] = spot
        self.cur["volume"] += max(volume, 0.0)
        self.cur["delta"] += signed
        self.pv += price * max(volume, 0.0)
        self.vol += max(volume, 0.0)
        self.cvd += signed
        self.prev_close = price

    def seed_bar(self, t0: float, o: float, h: float, l: float, c: float, volume: float, spot: float | None = None):
        signed = 0.0
        if self.prev_close is not None:
            signed = volume * (1.0 if c >= self.prev_close else -1.0)
        self.rows.append({
            "t0": int(t0),
            "open": o, "high": h, "low": l, "close": c,
            "spot": spot if spot else c,
            "volume": max(volume, 0.0),
            "delta": signed,
        })
        self.pv += ((h + l + c) / 3.0) * max(volume, 0.0)
        self.vol += max(volume, 0.0)
        self.cvd += signed
        self.prev_close = c
        if len(self.rows) > 240:
            self.rows = self.rows[-240:]

    def vwap(self) -> float | None:
        if self.vol <= 0:
            return self.cur["close"] if self.cur else None
        return self.pv / self.vol

    def as_frame(self) -> pd.DataFrame:
        rows = list(self.rows)
        if self.cur:
            rows.append(dict(self.cur))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["vwap"] = np.nan
        # reconstruct running vwap from stored bars is approximate; use live vwap on last
        df["typical"] = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_pv = (df["typical"] * df["volume"]).cumsum()
        cum_v = df["volume"].cumsum().replace(0, np.nan)
        df["vwap"] = cum_pv / cum_v
        df["vwap"] = df["vwap"].ffill()
        raw_efi = df["close"].diff().fillna(0.0) * df["volume"]
        df["efi13"] = raw_efi.ewm(span=13, adjust=False).mean()
        df["cvd"] = df["delta"].cumsum()
        df["spot_px"] = df["spot"]
        return df


BARS = SessionBars(BAR_SECONDS)


# ---------- VA (same models as the Streamlit playbook, no GEX) ----------
VA_PLAYBOOK = {
    "M1S_WATCH": ("Range · VAH probe", "WATCH SHORT (VAH fade)"),
    "M1S_ENTRY": ("Range · VAH mean-reversion SHORT", "SHORT MEAN-REVERSION (Target: POC → VAL)"),
    "M1S_ADD": ("Range · VAH failed acceptance", "ADD SHORT (Retest VAH from below)"),
    "M1L_WATCH": ("Range · VAL probe", "WATCH LONG (VAL bounce)"),
    "M1L_ENTRY": ("Range · VAL mean-reversion LONG", "LONG MEAN-REVERSION (Target: POC → VAH)"),
    "M1L_ADD": ("Range · VAL failed breakdown", "ADD LONG (Retest VAL from above)"),
    "M2L_WATCH": ("Trend · VAH break", "WATCH LONG (VAH acceptance)"),
    "M2L_ENTRY": ("Trend · VAH acceptance LONG", "LONG BREAKOUT (Hold above VAH / trail under box)"),
    "M2L_ADD": ("Trend · VAH retest from above", "ADD LONG (Retest old VAH / new box low)"),
    "M2S_WATCH": ("Trend · VAL break", "WATCH SHORT (VAL acceptance)"),
    "M2S_ENTRY": ("Trend · VAL acceptance SHORT", "SHORT BREAKDOWN (Hold below VAL / trail above box)"),
    "M2S_ADD": ("Trend · VAL retest from below", "ADD SHORT (Retest old VAL / new box high)"),
    "INSIDE": ("Inside value", "NO ENTRY"),
    "CHOP": ("Regime mixed / low efficiency", "NO ENTRY"),
}


def _value_area(df: pd.DataFrame) -> dict:
    if df.empty or df["volume"].sum() <= 0:
        return {}
    px = df["close"].astype(float)
    step = 5.0
    lo, hi = float(px.min()), float(px.max())
    bins = np.arange(lo, hi + step, step)
    if len(bins) < 3:
        poc = float(px.iloc[-1])
        return {"poc": poc, "vah": poc, "val": poc}
    idx = np.clip(np.digitize(px, bins) - 1, 0, len(bins) - 2)
    vol = np.zeros(len(bins) - 1)
    for i, v in zip(idx, df["volume"].astype(float)):
        vol[i] += v
    poc_i = int(vol.argmax())
    target = vol.sum() * 0.70
    taken = vol[poc_i]
    a, b = poc_i, poc_i
    while taken < target and (a > 0 or b < len(vol) - 1):
        left = vol[a - 1] if a > 0 else -1
        right = vol[b + 1] if b < len(vol) - 1 else -1
        if right >= left:
            b = min(b + 1, len(vol) - 1)
            taken += max(right, 0)
        else:
            a = max(a - 1, 0)
            taken += max(left, 0)
    mid = (bins[:-1] + bins[1:]) / 2.0
    return {"poc": float(mid[poc_i]), "val": float(bins[a]), "vah": float(bins[b + 1])}


def _eff(df: pd.DataFrame, n=20) -> float:
    c = df["close"].astype(float).tail(n)
    if len(c) < 8:
        return 0.0
    net = abs(float(c.iloc[-1] - c.iloc[0]))
    path = float(c.diff().abs().sum())
    return net / max(path, 1e-9)


def classify_va(df: pd.DataFrame) -> dict:
    empty = {"action": "NO ENTRY", "micro": "not enough bars", "model": "CHOP"}
    if df is None or len(df) < 12:
        return empty
    last = float(df["close"].iloc[-1])
    vp = _value_area(df)
    vah, val, poc = vp.get("vah"), vp.get("val"), vp.get("poc")
    eff = _eff(df)
    dens_inside = 0.0
    if vah and val:
        m = (df["close"] >= val) & (df["close"] <= vah)
        dens_inside = float(df.loc[m, "volume"].sum() / max(float(df["volume"].sum()), 1e-9))
    regime = "RANGE" if dens_inside >= 0.55 and eff < 0.28 else ("TREND" if eff >= 0.36 else "CHOP")
    std = float(df["close"].tail(20).std(ddof=1) or 1.0)
    loc = "INSIDE"
    if vah and last >= float(vah) + 0.35 * std:
        loc = "ABOVE_VAH"
    elif val and last <= float(val) - 0.35 * std:
        loc = "BELOW_VAL"
    elif vah and last > float(vah):
        loc = "VAH_EDGE"
    elif val and last < float(val):
        loc = "VAL_EDGE"
    efi = df["efi13"]
    px = df["close"].astype(float)
    efi_up = len(efi) and float(efi.iloc[-1]) > float(efi.tail(8).mean())
    efi_dn = len(efi) and float(efi.iloc[-1]) < float(efi.tail(8).mean())
    dv = float(df["delta"].tail(5).sum())
    code = "CHOP"
    if regime == "RANGE":
        if loc in ("ABOVE_VAH", "VAH_EDGE"):
            code = "M1S_ENTRY" if efi_dn and dv <= 0 else "M1S_WATCH"
        elif loc in ("BELOW_VAL", "VAL_EDGE"):
            code = "M1L_ENTRY" if efi_up and dv >= 0 else "M1L_WATCH"
        else:
            code = "INSIDE"
    elif regime == "TREND":
        if loc == "ABOVE_VAH":
            code = "M2L_ENTRY" if efi_up and dv > 0 else "M2L_WATCH"
        elif loc == "BELOW_VAL":
            code = "M2S_ENTRY" if efi_dn and dv < 0 else "M2S_WATCH"
        else:
            code = "CHOP"
    micro, action = VA_PLAYBOOK.get(code, VA_PLAYBOOK["CHOP"])
    note = f"{regime} {loc} eff={eff:.2f} dens={dens_inside:.2f} VAH {vah} VAL {val} POC {poc}"
    return {
        "action": action, "micro": f"{micro} · {note}", "model": code, "regime": regime,
        "vah": vah, "val": val, "poc": poc, "loc": loc,
    }


def _va_labels(df: pd.DataFrame) -> list:
    """Action when it changes, for vertical labels above candles."""
    out = []
    if df is None or len(df) < 12:
        return out
    prev = None
    step = max(1, len(df) // 40)
    for i in range(11, len(df), step):
        rec = classify_va(df.iloc[: i + 1])
        act = rec.get("action") or ""
        if act == prev or act == "NO ENTRY":
            continue
        prev = act
        ts = df.iloc[i].get("t0")
        out.append({"i": int(i), "t": str(ts), "action": act})
    rec = classify_va(df)
    if rec.get("action") and rec.get("action") != prev:
        out.append({"i": int(len(df) - 1), "t": str(df.iloc[-1].get("t0")), "action": rec["action"]})
    return out[-12:]


def apply_bar_state(spot, fut):
    df = BARS.as_frame()
    vwap = BARS.vwap()
    last = df.iloc[-1] if not df.empty else None
    va = classify_va(df) if not df.empty else {"action": "NO ENTRY", "micro": "warming up"}
    bars_out = []
    pdh = pdl = None
    if not df.empty:
        tail = df.tail(120).copy()
        # previous-day high/low from seeded t0 if present
        if "t0" in tail.columns:
            days = pd.to_datetime(tail["t0"], unit="s", errors="coerce")
            tail["_d"] = days.dt.date
            uniq = [d for d in tail["_d"].dropna().unique()]
            if len(uniq) >= 2:
                prevd = tail[tail["_d"] == uniq[-2]]
                if not prevd.empty:
                    pdh = float(prevd["high"].max())
                    pdl = float(prevd["low"].min())
        for _, r in tail.iterrows():
            t0 = r.get("t0")
            try:
                tstr = pd.to_datetime(t0, unit="s").strftime("%H:%M")
            except Exception:
                tstr = ""
            bars_out.append({
                "t": tstr,
                "open": float(r["open"]) if "open" in r and pd.notna(r["open"]) else float(r["close"]),
                "high": float(r["high"]) if "high" in r and pd.notna(r["high"]) else float(r["close"]),
                "low": float(r["low"]) if "low" in r and pd.notna(r["low"]) else float(r["close"]),
                "close": float(r["close"]),
                "volume": float(r["volume"]) if pd.notna(r.get("volume")) else 0.0,
                "delta": float(r["delta"]) if pd.notna(r.get("delta")) else 0.0,
                "vwap": float(r["vwap"]) if pd.notna(r.get("vwap")) else None,
                "efi13": float(r["efi13"]) if pd.notna(r.get("efi13")) else 0.0,
                "cvd": float(r["cvd"]) if pd.notna(r.get("cvd")) else 0.0,
            })
        va["labels"] = _va_labels(tail.reset_index(drop=True))
        va["pdh"] = pdh
        va["pdl"] = pdl
    with LOCK:
        SNAPSHOT.update({
            "spot": spot,
            "fut": fut,
            "vwap": vwap,
            "efi": float(last["efi13"]) if last is not None else None,
            "cvd": float(last["cvd"]) if last is not None else None,
            "bars": bars_out,
            "va": va,
        })
    _publish()


# ---------- Angel WS thread ----------
def angel_thread():
    from SmartApi import SmartConnect
    from SmartApi.smartWebSocketV2 import SmartWebSocketV2
    import pyotp

    api_key = os.getenv("API_KEY", "")
    client = os.getenv("CLIENT_CODE", "")
    pin = os.getenv("PIN", "")
    totp_secret = os.getenv("TOTP_SECRET", "")
    if not all([api_key, client, pin, totp_secret]):
        print("Missing API_KEY / CLIENT_CODE / PIN / TOTP_SECRET in .env")
        return

    api = SmartConnect(api_key=api_key)
    sess = api.generateSession(client, pin, pyotp.TOTP(totp_secret).now())
    if not sess or not sess.get("status"):
        print("SmartAPI login failed", sess)
        return
    jwt = sess["data"]["jwtToken"]
    feed = api.getfeedToken()
    sws = SmartWebSocketV2(jwt, api_key, client, feed)

    fut_token = resolve_nifty_fut_token()
    last_spot = {"px": None}
    last_fut = {"px": None, "vol": 0.0}

    def seed_history():
        """Load last session 3-min futures candles so charts/VA are not empty."""
        if not fut_token:
            print("no fut token — skip candle seed")
            return
        from datetime import datetime, timedelta
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=5)
        interval = "THREE_MINUTE" if BAR_SECONDS <= 180 else "FIVE_MINUTE"
        param = {
            "exchange": "NFO",
            "symboltoken": str(fut_token),
            "interval": interval,
            "fromdate": from_dt.strftime("%Y-%m-%d 09:15"),
            "todate": to_dt.strftime("%Y-%m-%d 15:30"),
        }
        try:
            res = api.getCandleData(param)
        except Exception as e:
            print("seed candles failed", e)
            return
        rows = (res or {}).get("data") or []
        print(f"seeded {len(rows)} {interval} candles")
        for row in rows:
            # [timestamp, open, high, low, close, volume]
            ts = pd.to_datetime(row[0])
            t0 = ts.timestamp()
            o, h, l, c = float(row[1]), float(row[2]), float(row[3]), float(row[4])
            vol = float(row[5] or 0)
            BARS.seed_bar(t0, o, h, l, c, vol, spot=c)
        if BARS.rows:
            last_fut["px"] = float(BARS.rows[-1]["close"])
            apply_bar_state(last_spot["px"], last_fut["px"])

    seed_history()

    def on_data(_ws, msg):
        try:
            token = str(msg.get("token") or "")
            ltp = msg.get("last_traded_price")
            if ltp is None:
                return
            px = float(ltp)
            # Angel often sends paise * 100 for indices; Nifty LTP ~2500000 → divide
            if px > 100000:
                px = px / 100.0
            vol = float(msg.get("last_traded_quantity") or msg.get("volume_trade_for_the_day") or 0)
            ts = float(msg.get("exchange_timestamp") or time.time() * 1000) / 1000.0
            if token == str(NIFTY_TOKEN):
                last_spot["px"] = px
                if not fut_token:
                    BARS.on_tick(ts, px, 0.0, spot=px)
            elif fut_token and token == str(fut_token):
                last_fut["px"] = px
                day_v = float(msg.get("volume_trade_for_the_day") or 0)
                inc = 0.0
                if day_v and last_fut["vol"] and day_v >= last_fut["vol"]:
                    inc = day_v - last_fut["vol"]
                last_fut["vol"] = day_v or last_fut["vol"]
                BARS.on_tick(ts, px, inc or max(vol, 0.0), spot=last_spot["px"] or px)
            apply_bar_state(last_spot["px"], last_fut["px"] or last_spot["px"])
        except Exception as e:
            print("tick parse", e)

    def on_open(_ws):
        tokens = [{"exchangeType": 1, "tokens": [str(NIFTY_TOKEN)]}]
        if fut_token:
            tokens.append({"exchangeType": FUT_EXCHANGE_TYPE, "tokens": [str(fut_token)]})
        sws.subscribe("tape01", 3, tokens)
        print("subscribed", tokens)

    sws.on_data = on_data
    sws.on_open = on_open
    print("connecting Angel SmartWebSocketV2…")
    sws.connect()


# ---------- tiny HTTP + WebSocket (stdlib handshake) ----------
import base64
import hashlib
import struct

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/live_tape.html"):
            data = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/ws":
            self._ws()
            return
        self.send_error(404)

    def log_message(self, fmt, *args):
        pass

    def _ws(self):
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self.send_error(400)
            return
        accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        q: asyncio.Queue = asyncio.Queue()
        CLIENTS.add(q)
        with LOCK:
            first = json.dumps(SNAPSHOT, default=str)
        try:
            self._send_text(first)
            while True:
                # block without asyncio: poll snapshot every 200ms if queue empty
                try:
                    # thread is not the event loop; just sleep-push latest
                    time.sleep(0.2)
                    with LOCK:
                        payload = json.dumps(SNAPSHOT, default=str)
                    self._send_text(payload)
                except (BrokenPipeError, ConnectionResetError):
                    break
        finally:
            CLIENTS.discard(q)

    def _send_text(self, text: str):
        raw = text.encode("utf-8")
        header = bytearray([0x81])
        n = len(raw)
        if n < 126:
            header.append(n)
        elif n < 65536:
            header.append(126)
            header.extend(struct.pack("!H", n))
        else:
            header.append(127)
            header.extend(struct.pack("!Q", n))
        self.wfile.write(header + raw)


def main():
    t = threading.Thread(target=angel_thread, daemon=True)
    t.start()
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"tape UI  http://{HOST}:{PORT}")
    print("GEX/VEX/IV are not loaded on this page.")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
