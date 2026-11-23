"""Trilingual alerts: the LLM (Groq by default, see llm.py) writes them from STRUCTURED FACTS ONLY; output is validated; fixed templates are the fallback."""
import json
import os
import re

LANGS = ("en", "hi", "gu")
MAX_LEN = 320
from . import llm as _llm
_DIG = {**{ord(c): str(i) for i, c in enumerate("०१२३४५६७८९")}, **{ord(c): str(i) for i, c in enumerate("૦૧૨૩૪૫૬૭૮૯")}}
_RANGES = {"hi": re.compile(r"[\u0900-\u097F]"), "gu": re.compile(r"[\u0A80-\u0AFF]")}


def make_facts(station_cfg, pred):
    """Facts handed to the LLM / templates. Only these may appear in the alert."""
    eta = pred.get("eta_minutes")
    return {
        "status": pred["status"],
        "name": {l: station_cfg.get(k, station_cfg["name"]) for l, k in (("en", "name"), ("hi", "name_hi"), ("gu", "name_gu"))},
        "current_level_cm": int(round(pred["current_cm"])),
        "danger_level_cm": int(round(pred["danger_cm"])),
        "eta_minutes": int(eta) if eta else None,
        "window_minutes": 180,
        "flood_probability_percent": int(round(pred["flood_probability"] * 100)),
        "alt_route": station_cfg.get("alt_route", {}),
    }


# ---------------- templates (always available) ----------------
def template_alert(f):
    n, lv, dg, pr = f["name"], f["current_level_cm"], f["danger_level_cm"], f["flood_probability_percent"]
    eta = f["eta_minutes"] or f["window_minutes"]
    alt = f["alt_route"]
    s = f["status"]
    if s == "danger":
        return {
            "en": f"DANGER: Water level at {n['en']} is {lv} cm, at or above the danger level of {dg} cm. {alt.get('en', '')}.",
            "hi": f"खतरा: {n['hi']} पर जलस्तर {lv} सेमी है, जो खतरे के स्तर {dg} सेमी तक पहुँच गया है। {alt.get('hi', '')}।",
            "gu": f"ખતરો: {n['gu']} પર પાણીનું સ્તર {lv} સેમી છે, જે ખતરાના સ્તર {dg} સેમી સુધી પહોંચી ગયું છે. {alt.get('gu', '')}.",
        }
    if s == "warning":
        return {
            "en": f"WARNING: Flood risk at {n['en']} within {eta} minutes. Level now {lv} cm (danger {dg} cm). Probability {pr}%. {alt.get('en', '')}.",
            "hi": f"चेतावनी: {n['hi']} पर अगले {eta} मिनट में बाढ़ का खतरा। अभी जलस्तर {lv} सेमी (खतरा स्तर {dg} सेमी)। संभावना {pr}%। {alt.get('hi', '')}।",
            "gu": f"ચેતવણી: {n['gu']} પર આગામી {eta} મિનિટમાં પૂરનું જોખમ. હાલનું સ્તર {lv} સેમી (ખતરાનું સ્તર {dg} સેમી). સંભાવના {pr}%. {alt.get('gu', '')}.",
        }
    return {
        "en": f"SAFE: Water level at {n['en']} is {lv} cm (danger level {dg} cm). No flood risk expected now.",
        "hi": f"सुरक्षित: {n['hi']} पर जलस्तर {lv} सेमी है (खतरा स्तर {dg} सेमी)। अभी बाढ़ का खतरा नहीं है।",
        "gu": f"સુરક્ષિત: {n['gu']} પર પાણીનું સ્તર {lv} સેમી છે (ખતરાનું સ્તર {dg} સેમી). હાલ પૂરનું જોખમ નથી.",
    }


# ---------------- validator ----------------
def _nums(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text.translate(_DIG)))


def allowed_numbers(f):
    nums = {str(f["current_level_cm"]), str(f["danger_level_cm"]), str(f["flood_probability_percent"]),
            str(f["window_minutes"])}
    if f["eta_minutes"]:
        nums.add(str(f["eta_minutes"]))
        nums.add(str(f["eta_minutes"] // 60))            # "within 2 hours"
    nums.add(str(f["window_minutes"] // 60))
    for v in list(f["alt_route"].values()) + list(f["name"].values()):
        nums |= _nums(v)
    return nums


def validate_alert(texts, f):
    """Returns list of error strings (empty = valid)."""
    errs = []
    if not isinstance(texts, dict):
        return ["not a dict"]
    for l in LANGS:
        t = texts.get(l)
        if not isinstance(t, str) or not t.strip():
            errs.append(f"missing language {l}")
            continue
        if len(t) > MAX_LEN:
            errs.append(f"{l} too long ({len(t)}>{MAX_LEN})")
        if l in _RANGES and not _RANGES[l].search(t):
            errs.append(f"{l} text is not in the right script")
        if l == "en" and re.search(r"[\u0900-\u097F\u0A80-\u0AFF]", t):
            errs.append("en contains Indic script")
        if f["name"][l] not in t:
            errs.append(f"{l} missing station name")
        n = _nums(t)
        if str(f["current_level_cm"]) not in n:
            errs.append(f"{l} missing current level")
        if f["status"] in ("warning", "danger") and str(f["danger_level_cm"]) not in n:
            errs.append(f"{l} missing danger level")
        extra = n - allowed_numbers(f)
        if extra:
            errs.append(f"{l} has numbers not in facts: {sorted(extra)}")
        r = f["alt_route"].get(l)
        if f["status"] in ("warning", "danger") and r and r not in t:
            errs.append(f"{l} missing alternative route")
    return errs


# ---------------- LLM (Groq) ----------------
SYSTEM = ("You write short public flood alerts for Vadodara in English, Hindi and Gujarati. "
          "Use ONLY the facts in the JSON the user sends. Never invent numbers, places, times or advice. "
          "Copy station names and the alternative route text exactly as given for each language. "
          "Write numbers with ASCII digits. Each message must be under 280 characters and include the current "
          "level and, for warning/danger, the danger level and the alternative route. For warning, include the "
          "time window (eta_minutes, or window_minutes if eta is null) and probability. "
          'Reply with ONLY JSON: {"en": "...", "hi": "...", "gu": "..."}')


def call_llm(f, timeout=20.0):
    """Groq by default; key from GROQ_API_KEY (env). Never hard-coded."""
    return _llm.chat(SYSTEM, json.dumps(f, ensure_ascii=False), max_tokens=900, json_mode=True, timeout=timeout)


call_claude = call_llm          # backwards-compatible name


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    return json.loads(text)


def generate_alert(f, llm=None):
    """Returns {"texts": {...}, "source": "llm"|"template", "errors": [...]}.
    llm: callable(facts)->str, default call_llm (used only if an LLM key is set)."""
    errors = []
    if llm is None and _llm.available():
        llm = call_llm
    if llm is not None:
        try:
            texts = parse_json(llm(f))
            errors = validate_alert(texts, f)
            if not errors:
                return {"texts": {l: texts[l].strip() for l in LANGS}, "source": "llm", "errors": []}
        except Exception as e:                         # network, auth, parse ... -> fallback
            errors = [f"llm failed: {type(e).__name__}: {e}"]
    else:
        errors = ["no LLM key (GROQ_API_KEY): using templates"]
    t = template_alert(f)
    t = {l: re.sub(r"\s+([.।])", r"\1", t[l]).replace(" .", ".") for l in LANGS}
    return {"texts": t, "source": "template", "errors": errors}
