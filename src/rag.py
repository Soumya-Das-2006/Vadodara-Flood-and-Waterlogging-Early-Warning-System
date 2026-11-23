"""Small RAG chat: retrieve live forecast + alert log for the stations in the question, then answer.
The LLM (Groq by default, if GROQ_API_KEY is set) answers ONLY from the retrieved context; otherwise a rule-based answer is built
from the same context, so the feature works offline."""
import json
import re
from . import alerts as A
from . import llm as _llm

ALIASES = {
    "akota": ["akota", "अकोटा", "અકોટા"],
    "alkapuri": ["alkapuri", "अलकापुरी", "અલકાપુરી"],
    "kamnath": ["kamnath", "vishwamitri", "कामनाथ", "विश्वामित्री", "કામનાથ", "વિશ્વામિત્રી"],
}
WHY = ["why", "reason", "क्यों", "कारण", "કેમ", "શા માટે", "કારણ"]
HIST = ["history", "alerts", "recent", "past", "इतिहास", "अलर्ट", "ઇતિહાસ", "ચેતવણીઓ"]


def detect_lang(q):
    if re.search(r"[\u0A80-\u0AFF]", q):
        return "gu"
    if re.search(r"[\u0900-\u097F]", q):
        return "hi"
    return "en"


def find_stations(q, ids):
    ql = q.lower()
    hit = [s for s in ids if any(a.lower() in ql for a in ALIASES.get(s, [s]))]
    return hit or list(ids)


def retrieve(q, db, cfg):
    sts = find_stations(q, cfg["stations"])
    ctx = []
    for s in sts:
        fc = db.latest_forecast(s)
        st = db.get_state(s)
        ctx.append({"station": s, "name": cfg["stations"][s]["name"], "state": st["status"], "latest_forecast": fc,
                    "recent_alerts": [{"id": a["id"], "time": a["ts"], "from": a["from_status"], "to": a["to_status"],
                                       "facts": a["facts"], "text_en": a["texts"].get("en"), "source": a["source"]}
                                      for a in db.alerts(limit=5, station=s)]})
    return ctx


SYS = ("You are the Vadodara flood early-warning assistant. Answer ONLY from the CONTEXT JSON. If the answer is not in "
       "the context, say you do not have that information. Never invent numbers, places or times. Reply in the same "
       "language as the question (English, Hindi or Gujarati), in at most 4 short sentences.")


def _rule_based(q, lang, ctx, cfg):
    ql = q.lower()
    c = ctx[0] if len(ctx) == 1 else None
    if any(w in ql for w in WHY):
        for item in ctx:
            if item["recent_alerts"]:
                a = item["recent_alerts"][0]; f = a["facts"]
                fc = item["latest_forecast"] or {}
                p50 = (fc.get("interval_cm", {}).get("3h", {}) or {}).get("p50")
                lv, dg, pr = f["current_level_cm"], f["danger_level_cm"], f["flood_probability_percent"]
                n = f["name"][lang]
                return {
                    "en": f"The alert for {n} changed from {a['from']} to {a['to']} at {a['time']}: level {lv} cm vs danger {dg} cm, flood probability {pr}%"
                          + (f", median 3 h forecast {round(p50)} cm." if p50 is not None else "."),
                    "hi": f"{n} के लिए अलर्ट {a['time']} पर {a['from']} से {a['to']} हुआ: जलस्तर {lv} सेमी, खतरा स्तर {dg} सेमी, बाढ़ की संभावना {pr}%।",
                    "gu": f"{n} માટે ચેતવણી {a['time']} એ {a['from']} થી {a['to']} થઈ: સ્તર {lv} સેમી, ખતરાનું સ્તર {dg} સેમી, પૂરની સંભાવના {pr}%.",
                }[lang]
        return {"en": "No alert has been sent yet, so there is nothing to explain.",
                "hi": "अभी तक कोई अलर्ट नहीं भेजा गया है।", "gu": "હજુ સુધી કોઈ ચેતવણી મોકલાઈ નથી."}[lang]
    if any(w in ql for w in HIST):
        lines = [f"{a['time']} {i['name']}: {a['from']} -> {a['to']}" for i in ctx for a in i["recent_alerts"]]
        return "\n".join(lines) if lines else {"en": "No alerts yet.", "hi": "अभी कोई अलर्ट नहीं।", "gu": "હજુ કોઈ ચેતવણી નથી."}[lang]
    out = []
    for item in ctx:
        fc = item["latest_forecast"]
        if not fc:
            out.append({"en": f"No readings yet for {item['name']}.", "hi": f"{item['name']} के लिए अभी कोई रीडिंग नहीं।",
                        "gu": f"{item['name']} માટે હજુ કોઈ રીડિંગ નથી."}[lang])
            continue
        pred = dict(fc, status=item["state"])
        f = A.make_facts(cfg["stations"][item["station"]], pred)
        out.append(A.template_alert(f)[lang])
    return "\n".join(out)


def answer(question, db, cfg, llm=None):
    lang = detect_lang(question)
    ctx = retrieve(question, db, cfg)
    if llm is None and _llm.available():
        llm = _ask_llm
    if llm is not None:
        try:
            txt = llm(question, ctx)
            if txt and txt.strip():
                return {"answer": txt.strip(), "source": "llm", "language": lang, "context": ctx}
        except Exception:
            pass
    return {"answer": _rule_based(question, lang, ctx, cfg), "source": "rules", "language": lang, "context": ctx}


def _ask_llm(question, ctx):
    return _llm.chat(SYS, f"CONTEXT:\n{json.dumps(ctx, ensure_ascii=False, default=str)}\n\nQUESTION: {question}", max_tokens=500)
