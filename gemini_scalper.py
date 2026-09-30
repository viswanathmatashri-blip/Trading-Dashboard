"""Gemini Scalper — TradingView CE / Spot / PE screenshot → watch setups."""
from __future__ import annotations

import base64
import json
import os
import time

import requests
import streamlit as st

MODELS = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)

PROMPT = """You are an Indian / MCX options scalper reading TradingView panes.
The image(s) are in this layout when combined: LEFT = ATM CALL (CE), CENTER = SPOT or FUTURES, RIGHT = ATM PUT (PE).
Each pane has candles, VWAP, visible-range volume profile, EFI (red) and OBV (blue).

BUY only. Never sell or write premium.
Give WATCH setups only — levels to wait for, not fills that need future bars.

Read from the chart pixels:
- last OHLC printed on each pane
- VWAP line vs last price
- volume-profile POC / HVN / LVN and the red/blue marked prices
- EFI slope (rising / flattening / falling) and zero-line
- OBV slope vs price
- whether CE and PE confirm or fight the futures pane

Return ONLY JSON:
{
  "instrument": "e.g. CRUDEOILM 15 OCT 8750",
  "read": {
    "spot": "one line: last vs VWAP vs VP node",
    "ce": "one line",
    "pe": "one line",
    "efi_obv": "one line on all three panes"
  },
  "CE": {
    "watch": true or false,
    "contract": "8750 CE",
    "trigger": "what must print on CE or futures before entry",
    "entry": "447-452",
    "target": 470,
    "tgt_pct": 4.5,
    "tgt_why": "VP node / prior swing / VWAP",
    "sl": 432,
    "sl_pct": -3.4,
    "sl_why": "named level on the CE pane",
    "rr": 1.3,
    "rationale": "3 short sentences from the screenshot"
  },
  "PE": { same keys },
  "bias": "LONG CE watch | SHORT PE watch | BOTH | NONE"
}
If a side has no edge set watch=false, contract=NO TRADE, rationale why from the image.
Do not invent prices that are not readable on the screenshot.
"""


def _key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def _part(upload):
    raw = upload.getvalue()
    mime = (upload.type or "image/png").split(";")[0]
    if mime not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
        mime = "image/png"
    return {"inline_data": {"mime_type": mime, "data": base64.b64encode(raw).decode("ascii")}}


def _call_gemini(parts):
    key = _key()
    if not key:
        return None, "No GEMINI_API_KEY in Streamlit secrets"
    errors = []
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2500},
    }
    txt = ""
    used = ""
    for model in MODELS:
        for ver in ("v1beta", "v1"):
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/{ver}/models/{model}:generateContent",
                    params={"key": key},
                    json=body,
                    timeout=60,
                )
                if r.status_code >= 400:
                    errors.append(f"{model} {ver} {r.status_code}")
                    continue
                js = r.json()
                blobs = (((js.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
                txt = "".join(p.get("text", "") for p in blobs).strip()
                if txt:
                    used = model
                    break
            except Exception as e:
                errors.append(f"{model} {e}"[:80])
        if txt:
            break
    if not txt:
        return None, " · ".join(errors[-6:]) or "Gemini empty"
    raw = txt
    if "```" in raw:
        chunk = raw.split("```", 2)[1]
        raw = chunk[4:] if chunk.lstrip().startswith("json") else chunk
    try:
        data = json.loads(raw)
    except Exception:
        a, b = raw.find("{"), raw.rfind("}")
        try:
            data = json.loads(raw[a:b + 1]) if a >= 0 and b > a else None
        except Exception:
            data = None
    if not isinstance(data, dict):
        return None, "JSON parse failed: " + txt[:240]
    data["_model"] = used
    return data, ""


def _cell(side, block):
    if not isinstance(block, dict):
        return f"<div style='color:#78909C;'>No {side} setup</div>"
    if not block.get("watch") or "NO TRADE" in str(block.get("contract") or "").upper():
        return (
            f"<div style='color:#FF8A80;font-size:13px;'><b>NO TRADE {side}</b><br>"
            f"<span style='color:#90A4AE;font-size:12px;'>{block.get('rationale') or ''}</span></div>"
        )
    col = "#69F0AE" if side == "CE" else "#FF8A80"
    return (
        f"<div style='font-size:13px;line-height:1.45;color:#ECEFF1;'>"
        f"<b style='color:{col};font-size:15px;'>{block.get('contract') or side}</b><br>"
        f"Trigger : {block.get('trigger') or '—'}<br>"
        f"Entry Zone : {block.get('entry') or '—'}<br>"
        f"Target : {block.get('target')} ({block.get('tgt_pct')} %)"
        f"<div style='color:#90A4AE;'>Tgt why : {block.get('tgt_why') or '—'}</div>"
        f"SL : {block.get('sl')} ({block.get('sl_pct')} %)"
        f"<div style='color:#90A4AE;'>SL why : {block.get('sl_why') or '—'}</div>"
        f"R:R : {block.get('rr')}<br>"
        f"<span style='color:#B0BEC5;'>Rationale : {block.get('rationale') or ''}</span>"
        f"</div>"
    )


def render_gemini_scalper(*_a, **_k):
    st.markdown(
        "<div style='font-size:1.12rem;font-weight:700;color:#69F0AE;'>Gemini Scalper</div>"
        "<div style='font-size:12px;color:#90A4AE;margin-bottom:8px;'>"
        "Upload TradingView panes: CE left · Spot/Fut centre · PE right — same layout as your screenshot. "
        "Gemini reads VWAP, volume profile, EFI, OBV and returns watch setups only."
        "</div>",
        unsafe_allow_html=True,
    )
    if not _key():
        st.error("Add GEMINI_API_KEY to Streamlit secrets.")
        return

    mode = st.radio(
        "Screenshot layout",
        ["One combined image (CE | Spot | PE)", "Three separate images"],
        horizontal=True,
        key="gs_shot_mode",
    )
    files = []
    if mode.startswith("One"):
        one = st.file_uploader("Combined screenshot", type=["png", "jpg", "jpeg", "webp"], key="gs_one")
        if one:
            files.append(("combined CE|SPOT|PE", one))
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            ce = st.file_uploader("CE pane", type=["png", "jpg", "jpeg", "webp"], key="gs_ce")
        with c2:
            sp = st.file_uploader("Spot / Fut pane", type=["png", "jpg", "jpeg", "webp"], key="gs_sp")
        with c3:
            pe = st.file_uploader("PE pane", type=["png", "jpg", "jpeg", "webp"], key="gs_pe")
        if ce:
            files.append(("CE pane", ce))
        if sp:
            files.append(("SPOT/FUT pane", sp))
        if pe:
            files.append(("PE pane", pe))

    note = st.text_input("Optional note (index, TF, strike)", key="gs_note", placeholder="CRUDEOILM 5m 8750 CE/PE")
    go = st.button("Read charts → watch setups", type="primary", disabled=not files, key="gs_go")

    if files:
        prev = st.columns(len(files))
        for i, (lab, f) in enumerate(files):
            with prev[i]:
                st.caption(lab)
                st.image(f, use_container_width=True)

    if go and files:
        parts = [{"text": PROMPT + (f"\nNOTE: {note}" if note else "")}]
        for lab, f in files:
            parts.append({"text": f"IMAGE LABEL: {lab}"})
            parts.append(_part(f))
        st.info("Gemini loading — reading screenshot(s)…")
        with st.spinner("Gemini working…"):
            data, err = _call_gemini(parts)
        st.session_state["_gs_vision"] = data
        st.session_state["_gs_vision_err"] = err
        st.session_state["_gs_vision_ts"] = time.strftime("%H:%M:%S")

    err = st.session_state.get("_gs_vision_err")
    data = st.session_state.get("_gs_vision")
    if err and not data:
        st.error(err)
        return
    if not data:
        st.caption("No setups yet — drop a screenshot and click Read charts.")
        return

    ts = st.session_state.get("_gs_vision_ts") or ""
    model = data.get("_model") or ""
    st.caption(f"Watch setups · {ts} · {model} · {data.get('instrument') or ''} · bias {data.get('bias') or '—'}")
    rd = data.get("read") or {}
    if rd:
        st.markdown(
            f"<div style='font-size:12px;color:#B0BEC5;line-height:1.4;margin:4px 0 10px;'>"
            f"<b>Spot</b> {rd.get('spot') or '—'}<br>"
            f"<b>CE</b> {rd.get('ce') or '—'}<br>"
            f"<b>PE</b> {rd.get('pe') or '—'}<br>"
            f"<b>EFI/OBV</b> {rd.get('efi_obv') or '—'}"
            f"</div>",
            unsafe_allow_html=True,
        )
    left, right = st.columns(2)
    with left:
        st.markdown("**CE watch**", unsafe_allow_html=True)
        st.markdown(_cell("CE", data.get("CE")), unsafe_allow_html=True)
    with right:
        st.markdown("**PE watch**", unsafe_allow_html=True)
        st.markdown(_cell("PE", data.get("PE")), unsafe_allow_html=True)
