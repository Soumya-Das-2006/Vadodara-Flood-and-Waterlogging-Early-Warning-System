import json
import pytest
from src import alerts as A
from src.config import load_config

@pytest.fixture
def facts():
    st = load_config()["stations"]["akota"]
    pred = {"status": "warning", "current_cm": 61.3, "danger_cm": 80, "flood_probability": 0.83, "eta_minutes": 120}
    return A.make_facts(st, pred)

def test_templates_pass_validator_for_all_statuses(facts):
    for s in ("safe", "warning", "danger"):
        facts["status"] = s
        assert A.validate_alert(A.template_alert(facts), facts) == []

def test_validator_catches_problems(facts):
    good = A.template_alert(facts)
    assert A.validate_alert({**good, "gu": ""}, facts)                              # missing language
    assert any("numbers not in facts" in e for e in A.validate_alert({**good, "en": good["en"] + " Expect 500 mm."}, facts))
    assert any("too long" in e for e in A.validate_alert({**good, "en": good["en"] + "x" * 400}, facts))
    assert any("script" in e for e in A.validate_alert({**good, "hi": good["en"]}, facts))
    wrong = good["en"].replace("61", "16")
    assert any("current level" in e for e in A.validate_alert({**good, "en": wrong}, facts))
    assert any("route" in e for e in A.validate_alert({**good, "en": good["en"].replace(facts["alt_route"]["en"], "Use the highway")}, facts))

def test_indic_digits_are_normalised(facts):
    t = A.template_alert(facts)
    t["hi"] = t["hi"].replace("61", "६१").replace("80", "८०")
    assert A.validate_alert(t, facts) == []

def test_llm_valid_output_is_used(facts):
    out = A.generate_alert(facts, llm=lambda f: "```json\n" + json.dumps(A.template_alert(f), ensure_ascii=False) + "\n```")
    assert out["source"] == "llm" and not out["errors"]

def test_llm_invented_number_falls_back(facts):
    bad = A.template_alert(facts); bad["en"] += " Depth 999 cm."
    out = A.generate_alert(facts, llm=lambda f: json.dumps(bad, ensure_ascii=False))
    assert out["source"] == "template" and out["errors"]
    assert A.validate_alert(out["texts"], facts) == []

def test_llm_exception_and_garbage_fall_back(facts):
    def boom(f): raise TimeoutError("api down")
    assert A.generate_alert(facts, llm=boom)["source"] == "template"
    assert A.generate_alert(facts, llm=lambda f: "not json")["source"] == "template"

def test_no_key_uses_template(facts, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert A.generate_alert(facts)["source"] == "template"

def test_api_key_never_hardcoded():
    import pathlib, re
    for p in pathlib.Path("src").rglob("*.py"):
        assert not re.search(r"(sk-ant-|gsk_)[A-Za-z0-9]", p.read_text(encoding="utf-8"))


def test_groq_request_shape_and_use(facts, monkeypatch):
    """Mocked Groq HTTP call: right URL/auth/JSON mode, and its output goes through the validator."""
    from unittest import mock
    monkeypatch.setenv("GROQ_API_KEY", "test-key"); monkeypatch.delenv("LLM_PROVIDER", raising=False)
    seen = {}
    def fake_post(url, json=None, headers=None, timeout=None):
        import json as j
        seen.update(url=url, body=json, auth=headers["Authorization"])
        r = mock.Mock(); r.raise_for_status = lambda: None
        r.json = lambda: {"choices": [{"message": {"content": j.dumps(A.template_alert(facts), ensure_ascii=False)}}]}
        return r
    with mock.patch("src.llm.requests.post", fake_post):
        out = A.generate_alert(facts)
    assert out["source"] == "llm" and seen["url"].startswith("https://api.groq.com/openai/v1")
    assert seen["auth"] == "Bearer test-key" and seen["body"]["response_format"] == {"type": "json_object"}

def test_groq_http_error_falls_back(facts, monkeypatch):
    from unittest import mock
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    def boom(*a, **k): raise ConnectionError("down")
    with mock.patch("src.llm.requests.post", boom):
        assert A.generate_alert(facts)["source"] == "template"
