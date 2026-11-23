"""Streamlit dashboard: clean white UI, English / Hindi / Gujarati, Noto Sans family."""
import json
import html
import os
import pathlib

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")
KEY = os.environ.get("API_KEY", "dev-key")
FIGS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "figures"
COLORS = {"safe": "#1a7f37", "warning": "#a16207", "danger": "#c5221f", "unknown": "#64748b"}

T = {
    "en": {"title": "Vadodara Flood & Waterlogging Early Warning", "overview": "Overview", "alerts": "Alert history",
           "model": "Model performance", "chat": "Ask the assistant", "lang": "Language", "safe": "Safe", "warning": "Warning",
           "danger": "Danger", "unknown": "No data", "level": "Level", "danger_lvl": "Danger level", "prob": "Flood probability (3 h)",
           "eta": "Danger ETA", "none": "none expected", "min": "min", "station": "Station", "history": "Observed level",
           "forecast": "Forecast (p50)", "band": "p10 - p90", "dl": "Danger level", "no_alerts": "No alerts yet.",
           "api_down": "Cannot reach the API. Is the backend running?", "ask": "Ask, e.g. 'Is Akota Bridge safe now?'",
           "refresh": "Auto-refresh (s)",
           "map": "Map", "reg": "Level forecast error on TEST (cm)", "fl": "Flood within 3 h on TEST", "cm": "cm",
           "nav": "Dashboard navigation", "stations": "Stations monitored", "iot": "IoT / Live data",
           "iot_status": "Live connection", "mqtt": "MQTT", "last_reading": "Last reading",
           "level": "Water level", "rain": "Rainfall", "battery": "Battery",
           "recent": "Recent sensor readings", "add_reading": "Add sensor reading",
           "save_reading": "Save reading", "reading_saved": "Reading saved successfully.",
           "reading_failed": "Reading could not be saved.", "http_ingest": "ESP32 HTTP ingest",
           "setup_hint": "Leave MQTT_HOST empty in the ESP32 configuration and set API_URL and API_KEY.",
           "no_readings": "No sensor readings have been received for this station."},
    "hi": {"title": "वडोदरा बाढ़ और जलभराव पूर्व-चेतावनी", "overview": "अवलोकन", "alerts": "अलर्ट इतिहास",
           "model": "मॉडल प्रदर्शन", "chat": "सहायक से पूछें", "lang": "भाषा", "safe": "सुरक्षित", "warning": "चेतावनी",
           "danger": "खतरा", "unknown": "कोई डेटा नहीं", "level": "जलस्तर", "danger_lvl": "खतरा स्तर", "prob": "बाढ़ की संभावना (3 घंटे)",
           "eta": "खतरे तक समय", "none": "अपेक्षित नहीं", "min": "मिनट", "station": "स्टेशन", "history": "मापा गया स्तर",
           "forecast": "पूर्वानुमान (p50)", "band": "p10 - p90", "dl": "खतरा स्तर", "no_alerts": "अभी कोई अलर्ट नहीं।",
           "api_down": "API से संपर्क नहीं हो सका। क्या बैकएंड चालू है?", "ask": "पूछें, जैसे 'क्या अकोटा ब्रिज अभी सुरक्षित है?'",
           "refresh": "ऑटो-रिफ्रेश (सेकंड)",
           "map": "नक्शा", "reg": "TEST पर स्तर पूर्वानुमान त्रुटि (सेमी)", "fl": "TEST पर 3 घंटे में बाढ़", "cm": "सेमी",
           "nav": "डैशबोर्ड नेविगेशन", "stations": "निगरानी वाले स्टेशन", "iot": "IoT / लाइव डेटा",
           "iot_status": "लाइव कनेक्शन", "mqtt": "MQTT", "last_reading": "अंतिम रीडिंग",
           "level": "जलस्तर", "rain": "बारिश", "battery": "बैटरी",
           "recent": "हाल की सेंसर रीडिंग", "add_reading": "सेंसर रीडिंग जोड़ें",
           "save_reading": "रीडिंग सेव करें", "reading_saved": "रीडिंग सफलतापूर्वक सेव हो गई।",
           "reading_failed": "रीडिंग सेव नहीं हो सकी।", "http_ingest": "ESP32 HTTP इंटेज़्ट",
           "setup_hint": "ESP32 कॉन्फ़िगरेशन में MQTT_HOST खाली रखें और API_URL तथा API_KEY सेट करें।",
           "no_readings": "इस स्टेशन के लिए अभी कोई सेंसर रीडिंग नहीं मिली।"},
    "gu": {"title": "વડોદરા પૂર અને જળભરાવ પૂર્વ-ચેતવણી", "overview": "ઝાંખી", "alerts": "ચેતવણી ઇતિહાસ",
           "model": "મોડેલ પ્રદર્શન", "chat": "સહાયકને પૂછો", "lang": "ભાષા", "safe": "સુરક્ષિત", "warning": "ચેતવણી",
           "danger": "ખતરો", "unknown": "ડેટા નથી", "level": "સ્તર", "danger_lvl": "ખતરાનું સ્તર", "prob": "પૂરની સંભાવના (3 કલાક)",
           "eta": "ખતરા સુધીનો સમય", "none": "અપેક્ષિત નથી", "min": "મિનિટ", "station": "સ્ટેશન", "history": "માપેલું સ્તર",
           "forecast": "આગાહી (p50)", "band": "p10 - p90", "dl": "ખતરાનું સ્તર", "no_alerts": "હજુ કોઈ ચેતવણી નથી.",
           "api_down": "API સુધી પહોંચી શકાયું નથી. શું બેકએન્ડ ચાલુ છે?", "ask": "પૂછો, જેમ કે 'શું અકોટા બ્રિજ હાલ સુરક્ષિત છે?'",
           "refresh": "ઓટો-રિફ્રેશ (સેકન્ડ)",
           "map": "નકશો", "reg": "TEST પર સ્તર આગાહી ભૂલ (સેમી)", "fl": "TEST પર 3 કલાકમાં પૂર", "cm": "સેમી",
           "nav": "ડેશબોર્ડ નેવિગેશન", "stations": "નિરીક્ષણ હેઠળનાં સ્ટેશન", "iot": "IoT / લાઇવ ડેટા",
           "iot_status": "લાઇવ કનેક્શન", "mqtt": "MQTT", "last_reading": "છેલ્લું રીડિંગ",
           "level": "પાણીનું સ્તર", "rain": "વર્ષા", "battery": "બેટરી",
           "recent": "તાજેતરની સેન્સર રીડિંગ", "add_reading": "સેન્સર રીડિંગ ઉમેરો",
           "save_reading": "રીડિંગ સંગ્રહ કરો", "reading_saved": "રીડિંગ સફળતાપૂર્વક સંગ્રહ થઈ.",
           "reading_failed": "રીડিং સંગ્રહ થઈ નથી.", "http_ingest": "ESP32 HTTP ઇન્ટેજ",
           "setup_hint": "ESP32 કૉન્ફિગરેશનમાં MQTT_HOST ખાલી રાખો અને API_URL તથા API_KEY સેટ કરો.",
           "no_readings": "આ સ્ટેશન માટે હજુ કોઈ સેન્સર રીડિંગ પ્રાપ્ત નથી."},
}

CSS = """
<style>
:root {
  --ink: #15263b;
  --muted: #64748b;
  --line: #e2e8f0;
  --surface: #ffffff;
  --canvas: #f3f6fb;
  --accent: #2563eb;
}
html, body, [class*="css"], .stApp, button, input, textarea {
  font-family: Inter, "Noto Sans", "Noto Sans Devanagari", "Noto Sans Gujarati", sans-serif !important;
}
.stApp { background: var(--canvas); color: var(--ink); }
[data-testid="stHeader"] { background: rgba(243, 246, 251, .92); }
[data-testid="stToolbar"] { right: 1.5rem; }
.block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
section[data-testid="stSidebar"] {
  background: #10233b;
  border-right: 1px solid #203955;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  color: #d7e3f1 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
  background: #1b3552;
  color: #f8fafc;
  border-color: #36516f;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
  background: transparent;
  border-radius: 10px;
  padding: .35rem .5rem;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background: #1b3552;
}
.brand { padding: .5rem 0 1.4rem; border-bottom: 1px solid #2a435f; margin-bottom: 1.2rem; }
.brand-mark {
  display: inline-flex; align-items: center; justify-content: center;
  width: 38px; height: 38px; border-radius: 12px;
  color: #fff; background: #2563eb; font-size: 1.2rem; font-weight: 700;
}
.brand-name { color: #f8fafc; font-size: 1rem; font-weight: 700; margin-left: .55rem; }
.brand-caption { color: #9fb3c9; font-size: .77rem; margin-top: .55rem; }
.st-key-dashboard_hero {
  padding: 1.6rem 1.8rem;
  margin-bottom: .8rem;
  border: 1px solid #dce6f2;
  border-radius: 20px;
  background: linear-gradient(115deg, #fff 0%, #f8fbff 65%, #eaf2ff 100%);
  box-shadow: 0 8px 26px rgba(15, 35, 59, .05);
}
.st-key-dashboard_hero h1 {
  color: var(--ink); font-size: clamp(1.7rem, 3vw, 2.45rem);
  line-height: 1.15; margin: .8rem 0 .15rem;
}
.station-card {
  --status-color: #94a3b8;
  height: 100%; min-height: 210px; padding: 1.2rem 1.25rem;
  border: 1px solid var(--line); border-top: 4px solid var(--status-color);
  border-radius: 16px; background: var(--surface);
  box-shadow: 0 5px 18px rgba(15, 35, 59, .045);
  transition: transform .18s ease, box-shadow .18s ease;
}
.station-card:hover { transform: translateY(-2px); box-shadow: 0 10px 24px rgba(15, 35, 59, .09); }
.station-card h3 { color: var(--ink); font-size: 1.02rem; line-height: 1.35; min-height: 2.7rem; margin: 0 0 .7rem; }
.status-pill {
  display: inline-block; border-radius: 999px; padding: .25rem .68rem;
  color: #fff; background: var(--status-color);
  font-size: .76rem; font-weight: 700;
}
.station-level { color: var(--ink); font-size: 2rem; font-weight: 750; line-height: 1.15; margin-top: .65rem; }
.station-level-caption { color: var(--muted); font-size: .77rem; margin: .15rem 0 .7rem; }
.station-detail { display: flex; justify-content: space-between; gap: .6rem; padding: .28rem 0; border-top: 1px solid #f0f3f7; font-size: .8rem; }
.station-detail span { color: var(--muted); }
.station-detail strong { color: var(--ink); text-align: right; }
[data-testid="stMetric"] {
  padding: .9rem 1rem; border: 1px solid var(--line); border-radius: 14px;
  background: #fff; box-shadow: 0 4px 14px rgba(15, 35, 59, .035);
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: var(--ink); }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
  border: 1px solid var(--line); border-radius: 14px; background: #fff; padding: .35rem;
}
[data-testid="stExpander"] { border-color: var(--line); border-radius: 12px; }
h2, h3 { color: var(--ink); }
@media (max-width: 800px) {
  .block-container { padding: 1rem 1rem 2rem; }
  .st-key-dashboard_hero { padding: 1.25rem; border-radius: 15px; }
  .station-card { min-height: 0; }
}
</style>
"""


def api(path, method="get", **kw):
    try:
        r = requests.request(method, API + path, headers={"X-API-Key": KEY}, timeout=8, **kw)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def name_of(s, lang):
    return {"en": s["name"], "hi": s.get("name_hi") or s["name"], "gu": s.get("name_gu") or s["name"]}[lang]


def card(s, t, lang):
    status = s.get("status", "unknown")
    col = COLORS.get(status, COLORS["unknown"])
    status_text = html.escape(t.get(status, t["unknown"]))
    eta_minutes = s.get("eta_minutes")
    eta = f"{eta_minutes} {t['min']}" if eta_minutes is not None else t["none"]
    probability = s.get("flood_probability")
    prob = f"{round(probability * 100)}%" if probability is not None else "-"
    lvl = f"{s['current_cm']:.0f} {t['cm']}" if s.get("current_cm") is not None else "-"
    name = html.escape(name_of(s, lang))
    danger = html.escape(str(s.get("danger_cm", "-")))
    st.markdown(
        f"<article class='station-card' style='--status-color:{col}' aria-label='{name}'>"
        f"<h3>{name}</h3><span class='status-pill'>{status_text}</span>"
        f"<div class='station-level'>{lvl}</div><div class='station-level-caption'>{t['level']}</div>"
        f"<div class='station-detail'><span>{t['danger_lvl']}</span><strong>{danger} {t['cm']}</strong></div>"
        f"<div class='station-detail'><span>{t['prob']}</span><strong>{prob}</strong></div>"
        f"<div class='station-detail'><span>{t['eta']}</span><strong>{html.escape(eta)}</strong></div>"
        f"</article>",
        unsafe_allow_html=True,
    )


def chart(sid, t, danger):
    rd = api("/readings", params={"station": sid, "limit": 400})
    fc = api(f"/forecast/{sid}")
    fig = go.Figure()
    if rd:
        d = pd.DataFrame(rd); d["ts"] = pd.to_datetime(d["ts"])
        fig.add_trace(go.Scatter(x=d["ts"], y=d["level_cm"], name=t["history"], line=dict(color="#111827", width=2)))
        fig.add_trace(go.Bar(x=d["ts"], y=d["rain_mmh"], name="rain mm/h", yaxis="y2", marker_color="rgba(59,130,246,.35)"))
    if fc:
        t0 = pd.to_datetime(fc["as_of"])
        xs = [t0] + [t0 + pd.Timedelta(hours=h) for h in (1, 2, 3)]
        p50 = [fc["current_cm"]] + [fc["interval_cm"][f"{h}h"]["p50"] for h in (1, 2, 3)]
        lo = [fc["current_cm"]] + [fc["interval_cm"][f"{h}h"]["p10"] for h in (1, 2, 3)]
        hi = [fc["current_cm"]] + [fc["interval_cm"][f"{h}h"]["p90"] for h in (1, 2, 3)]
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], fill="toself", fillcolor="rgba(37,99,235,.18)",
                                 line=dict(width=0), name=t["band"], hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=p50, name=t["forecast"], line=dict(color="#2563eb", dash="dot", width=3)))
    fig.add_hline(y=danger, line_color="#dc2626", line_dash="dash", annotation_text=t["dl"])
    fig.update_layout(
        height=360, margin=dict(l=16, r=16, t=20, b=16), plot_bgcolor="#fff", paper_bgcolor="#fff",
        font=dict(color="#475569"), hovermode="x unified",
        yaxis=dict(title=t["cm"], gridcolor="#edf1f6", zerolinecolor="#edf1f6"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, range=[0, 120]),
        legend=dict(orientation="h", y=1.12), xaxis=dict(gridcolor="#edf1f6"),
    )
    st.plotly_chart(fig, use_container_width=True)


def page_overview(stations, t, lang):
    counts = {status: sum(s.get("status") == status for s in stations)
              for status in ("safe", "warning", "danger")}
    metric_cols = st.columns(4)
    metric_cols[0].metric(t["stations"], len(stations))
    for col, status in zip(metric_cols[1:], ("safe", "warning", "danger")):
        col.metric(t[status], counts[status])
    st.write("")
    cols = st.columns(len(stations))
    for c, s in zip(cols, stations):
        with c:
            card(s, t, lang)
    st.write("")
    st.subheader(t["history"])
    left, right = st.columns([3, 2])
    with left:
        ids = {name_of(s, lang): s for s in stations}
        pick = st.selectbox(t["station"], list(ids))
        chart(ids[pick]["id"], t, ids[pick]["danger_cm"])
    with right:
        st.subheader(t["map"])
        try:
            import folium
            from streamlit_folium import st_folium
            m = folium.Map(location=[22.30, 73.18], zoom_start=12, tiles="OpenStreetMap")
            for s in stations:
                folium.CircleMarker([s["lat"], s["lon"]], radius=12, color=COLORS[s["status"]], fill=True, fill_opacity=.85,
                                    tooltip=f"{name_of(s, lang)}: {t[s['status']]}").add_to(m)
            st_folium(m, height=380, use_container_width=True, returned_objects=[])
        except Exception as e:
            st.warning(f"map unavailable: {e}")


def page_alerts(t, lang):
    rows = api("/alerts", params={"limit": 50})
    if not rows:
        st.info(t["no_alerts"]); return
    for a in rows:
        col = COLORS.get(a["to_status"], "#888")
        st.markdown(f"<span class='chip' style='background:{col}'>{t[a['to_status']]}</span> "
                    f"**{a['station']}** &nbsp; <span class='muted'>{a['ts']} · {a['from_status']} → {a['to_status']} · "
                    f"{a['source']} · telegram: {a.get('notify') or '-'}</span>", unsafe_allow_html=True)
        st.write(a["texts"].get(lang) or a["texts"]["en"])
        with st.expander("EN / हिन्दी / ગુજરાતી"):
            for l in ("en", "hi", "gu"):
                st.write(a["texts"][l])
        st.divider()


def page_model(t):
    m = api("/model/metrics")
    if not m:
        st.info("metrics.json not found. Run `make train`."); return
    st.subheader(t["reg"])
    rows = [{"model": k, **{f"MAE {h}": v[h]["mae"] for h in ("1h", "2h", "3h")}, "R2 3h": v["3h"]["r2"]}
            for k, v in m["regression_test"].items()]
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
    st.subheader(t["fl"])
    keys = ["precision", "recall", "f1", "pr_auc", "roc_auc", "brier", "false_alarm_ratio", "event_recall", "lead_mean_min"]
    rows = [{"model": k, **{x: v.get(x) for x in keys}} for k, v in m["flood_test"].items()]
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
    for f in ("shap_summary.png", "pr_curves.png", "reliability.png", "test_event_forecast.png"):
        if (FIGS / f).exists():
            st.image(str(FIGS / f))


def page_chat(t, lang):
    st.subheader(t["chat"])
    st.session_state.setdefault("chat", [])

    suggestions = {
        "en": ["Is Akota Bridge safe now?", "Why was the alert sent?", "Show the recent alert history"],
        "hi": ["अकोटा ब्रिज अभी सुरक्षित है?", "अलर्ट क्यों भेजा गया?", "हाल के अलर्ट दिखाएं"],
        "gu": ["અકોટા બ્રિજ સુરક્ષિત છે?", "ચેતવણી શા માટે મોકલવામાં આવી?", "તાજેતરની चेतવणी ઇતિહાસ બતાવો"],
    }
    cols = st.columns(len(suggestions[lang]))
    question = None
    for col, suggestion in zip(cols, suggestions[lang]):
        if col.button(suggestion, use_container_width=True, key=f"chat_suggestion_{suggestion}"):
            question = suggestion

    if st.session_state["chat"]:
        if st.button("Clear chat", key="clear_chat"):
            st.session_state["chat"] = []
            st.rerun()

    for role, msg in st.session_state["chat"]:
        if role == "assistant":
            if isinstance(msg, tuple):
                source, text = msg
            else:
                source, text = "unknown", msg
            st.chat_message(role).write(text)
            st.caption(f"Answer source: {source}")
        else:
            st.chat_message(role).write(msg)

    q = question or st.chat_input(t["ask"])
    if q:
        st.session_state["chat"].append(("user", q)); st.chat_message("user").write(q)
        r = api("/chat", method="post", json={"question": q})
        if r:
            a = r["answer"]
            source = r.get("source", "unknown")
            st.session_state["chat"].append(("assistant", (source, a)))
            st.chat_message("assistant").write(a)
            st.caption(f"Answer source: {source}")
        else:
            a = t["api_down"]
            st.session_state["chat"].append(("assistant", ("unknown", a)))
            st.chat_message("assistant").write(a)


def page_iot(stations, t, lang):
    try:
        health = requests.get(API + "/health", timeout=8)
        health.raise_for_status()
        health = health.json()
    except Exception:
        health = {}

    status = health.get("status", "degraded")
    mqtt_connected = bool(health.get("mqtt_connected"))
    cols = st.columns(4)
    cols[0].metric(t["iot_status"], "Connected" if status == "ok" else "Unavailable")
    cols[1].metric(t["mqtt"], "Connected" if mqtt_connected else "Not configured")
    cols[2].metric(t["stations"], health.get("stations", len(stations)))
    cols[3].metric(t["last_reading"], health.get("last_reading") or "-")

    st.subheader(t["recent"])
    station_names = {name_of(s, lang): s for s in stations}
    selected_name = st.selectbox(t["station"], list(station_names))
    station = station_names[selected_name]
    readings = api("/readings", params={"station": station["id"], "limit": 500})

    if readings:
        frame = pd.DataFrame(readings)
        frame["ts"] = pd.to_datetime(frame["ts"])
        frame = frame.sort_values("ts")

        latest = frame.iloc[-1]
        metric_cols = st.columns(3)
        metric_cols[0].metric(t["level"], f"{latest['level_cm']:.1f} cm")
        metric_cols[1].metric(t["rain"], f"{latest['rain_mmh']:.1f} mm/h")
        metric_cols[2].metric(t["battery"], f"{latest['battery_v']:.2f} V" if pd.notna(latest.get("battery_v")) else "-")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frame["ts"], y=frame["level_cm"], name=t["level"],
                                 line=dict(color="#2563eb", width=2)))
        fig.add_hline(y=station["danger_cm"], line_color="#dc2626", line_dash="dash",
                      annotation_text=t["danger_lvl"])
        fig.update_layout(height=330, margin=dict(l=16, r=16, t=20, b=16),
                          plot_bgcolor="#fff", paper_bgcolor="#fff",
                          yaxis=dict(title=t["cm"], gridcolor="#edf1f6"),
                          xaxis=dict(gridcolor="#edf1f6"))
        st.plotly_chart(fig, use_container_width=True)

        display = frame.tail(100).sort_values("ts", ascending=False)
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.download_button(
            "Download CSV",
            display.to_csv(index=False).encode("utf-8"),
            file_name=f"{station['id']}_readings.csv",
            mime="text/csv",
        )
    else:
        st.info(t["no_readings"])

    st.subheader(t["add_reading"])
    with st.form("iot_reading_form", clear_on_submit=True):
        level = st.number_input(t["level"], min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
        rain = st.number_input(t["rain"], min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
        battery = st.number_input(t["battery"], min_value=0.0, max_value=20.0, value=4.0, step=0.01)
        submitted = st.form_submit_button(t["save_reading"])
        if submitted:
            body = {
                "station": station["id"],
                "level_cm": float(level),
                "rain_mmh": float(rain),
                "battery_v": float(battery),
                "ts": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
            }
            result = api("/ingest", method="post", json=body)
            if result and result.get("ok"):
                st.success(t["reading_saved"])
                st.rerun()
            else:
                st.error(t["reading_failed"])

    with st.expander(t["http_ingest"]):
        st.code(
            "# firmware/esp32/include/config.h\n"
            '#define STATION_ID "akota"\n'
            '#define MQTT_HOST ""\n'
            '#define API_URL "https://your-backend-host"\n'
            '#define API_KEY "YOUR_API_KEY"',
            language="cpp",
        )
        st.caption(t["setup_hint"])


def main():
    st.set_page_config(page_title="Vadodara Flood EWS", page_icon="🌊", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div class='brand'><span class='brand-mark'>V</span>"
        "<span class='brand-name'>Vadodara EWS</span>"
        "<div class='brand-caption'>Flood &amp; waterlogging monitoring</div></div>",
        unsafe_allow_html=True,
    )
    lang = st.sidebar.selectbox("Language / भाषा / ભાષા", ["en", "hi", "gu"],
                                format_func=lambda x: {"en": "English", "hi": "हिन्दी", "gu": "ગુજરાતી"}[x])
    t = T[lang]
    page = st.sidebar.radio(
        t["nav"], ["overview", "iot", "alerts", "model", "chat"],
        format_func=lambda key: t[key], label_visibility="collapsed",
    )
    every = st.sidebar.number_input(t["refresh"], 0, 60, 10)
    if every:
        try:
            from streamlit_autorefresh import st_autorefresh
            if page in ("overview", "iot", "alerts"):
                st_autorefresh(interval=int(every * 1000), key="auto")
        except Exception:
            pass
    with st.container(key="dashboard_hero"):
        st.title(t["title"])
    stations = api("/stations")
    if stations is None:
        st.error(t["api_down"]); return
    {"overview": lambda: page_overview(stations, t, lang), "iot": lambda: page_iot(stations, t, lang),
     "alerts": lambda: page_alerts(t, lang), "model": lambda: page_model(t),
     "chat": lambda: page_chat(t, lang)}[page]()


main()
