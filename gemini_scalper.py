"""Gemini Scalper — TradingView CE / Spot / PE screenshot → watch setups."""
from __future__ import annotations

import base64
import io
import json
import os
import time

import requests
import streamlit as st
import streamlit.components.v1 as components

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


def _from_data_url(s):
    s = (s or "").strip()
    if not s.startswith("data:image"):
        return None
    try:
        head, b64 = s.split(",", 1)
        raw = base64.b64decode(b64)
        mime = "image/png"
        if "image/jpeg" in head:
            mime = "image/jpeg"
        elif "image/webp" in head:
            mime = "image/webp"
        bio = io.BytesIO(raw)
        bio.name = "paste.png" if "png" in mime else "paste.jpg"
        bio.type = mime
        return bio
    except Exception:
        return None


class _PasteFile:
    def __init__(self, bio):
        self._bio = bio
        self.name = getattr(bio, "name", "paste.png")
        self.type = getattr(bio, "type", "image/png")

    def getvalue(self):
        self._bio.seek(0)
        return self._bio.read()


def _paste_dock(slot_key, hint):
    components.html(
        f"""
<div id="dock" style="border:1px dashed #546E7A;border-radius:8px;min-height:88px;
  background:#111418;color:#B0BEC5;font-family:sans-serif;padding:14px 16px;
  outline:none;cursor:text;" tabindex="0">
  <div style="font-size:13px;color:#ECEFF1;font-weight:600;">Click here, then paste (Ctrl+V / Cmd+V)</div>
  <div style="font-size:12px;margin-top:4px;">{hint}</div>
  <div id="st" style="font-size:12px;color:#69F0AE;margin-top:8px;"></div>
</div>
<script>
const SLOT = {json.dumps(slot_key)};
const dock = document.getElementById("dock");
dock.focus();
function writeParent(dataUrl) {{
  const doc = window.parent.document;
  const boxes = [...doc.querySelectorAll("textarea")];
  const ta = boxes.find(t => (t.getAttribute("aria-label") || "").indexOf(SLOT) >= 0)
          || boxes.find(t => (t.value || "").startsWith("data:image") === false && (t.getAttribute("aria-label") || "").indexOf("gs_paste") >= 0);
  const target = boxes.find(t => (t.getAttribute("aria-label") || "") === SLOT) || ta;
  if (!target) {{
    document.getElementById("st").textContent = "Paste captured — click Read charts after Streamlit picks it up.";
    return;
  }}
  const proto = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value");
  if (proto && proto.set) proto.set.call(target, dataUrl);
  else target.value = dataUrl;
  target.dispatchEvent(new Event("input", {{ bubbles: true }}));
  target.dispatchEvent(new Event("change", {{ bubbles: true }}));
  document.getElementById("st").textContent = "Pasted. If preview is empty, click Read charts once.";
}}
function grab(e) {{
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (const it of items) {{
    if (it.type && it.type.indexOf("image") === 0) {{
      e.preventDefault();
      const file = it.getAsFile();
      const reader = new FileReader();
      reader.onload = () => writeParent(reader.result);
      reader.readAsDataURL(file);
      document.getElementById("st").textContent = "Reading clipboard…";
      return;
    }}
  }}
}}
dock.addEventListener("paste", grab);
window.addEventListener("paste", grab);
document.addEventListener("paste", grab);
</script>
        """,
        height=120,
    )
    raw = st.text_area(slot_key, key=slot_key, height=1)
    return _from_data_url(raw)


def render_gemini_scalper(*_a, **_k):
    st.markdown(
        "<div style='font-size:1.12rem;font-weight:700;color:#69F0AE;'>Gemini Scalper</div>"
        "<div style='font-size:12px;color:#90A4AE;margin-bottom:8px;'>"
        "Paste TradingView panes (CE left · Spot/Fut centre · PE right). "
        "Click the box, then Ctrl+V / Cmd+V — no file upload."
        "</div>"
        "<style>div[data-testid='stTextArea'] textarea { min-height: 0 !important; height: 0 !important; "
        "opacity: 0; position: absolute; }</style>",
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
        one = _paste_dock("gs_paste_one", "Combined CE | Spot | PE screenshot")
        if one:
            files.append(("combined CE|SPOT|PE", _PasteFile(one)))
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            ce = _paste_dock("gs_paste_ce", "CE pane")
        with c2:
            sp = _paste_dock("gs_paste_sp", "Spot / Fut pane")
        with c3:
            pe = _paste_dock("gs_paste_pe", "PE pane")
        if ce:
            files.append(("CE pane", _PasteFile(ce)))
        if sp:
            files.append(("SPOT/FUT pane", _PasteFile(sp)))
        if pe:
            files.append(("PE pane", _PasteFile(pe)))

    note = st.text_input("Optional note (index, TF, strike)", key="gs_note", placeholder="CRUDEOILM 5m 8750 CE/PE")
    go = st.button("Read charts → watch setups", type="primary", disabled=not files, key="gs_go")

    if files:
        prev = st.columns(len(files))
        for i, (lab, f) in enumerate(files):
            with prev[i]:
                st.caption(lab)
                st.image(f.getvalue(), use_container_width=True)

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


ORDER = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "GOLDM", "CRUDEOIL"]


def seed_gemini_scalper(*_a, **_k):
    return {}
