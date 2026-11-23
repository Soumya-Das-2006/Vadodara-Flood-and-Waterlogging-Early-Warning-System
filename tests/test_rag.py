import pandas as pd
from backend.app.db import DB
from backend.app.service import FloodService
from src import rag
from tests.test_service import FakePred, _svc

def test_status_and_why_in_three_languages(cfg):
    svc = _svc(FakePred(["warning"] * 3), cfg); svc.evaluate("akota")
    en = rag.answer("Is Akota Bridge safe now?", svc.db, cfg)
    assert en["language"] == "en" and "WARNING" in en["answer"] and en["source"] == "rules"
    why = rag.answer("Why was the alert sent for Akota?", svc.db, cfg)["answer"]
    assert "safe to warning" in why and "80" in why
    hi = rag.answer("अकोटा ब्रिज अभी सुरक्षित है?", svc.db, cfg)
    assert hi["language"] == "hi" and "चेतावनी" in hi["answer"]
    gu = rag.answer("અકોટા બ્રિજ સુરક્ષિત છે?", svc.db, cfg)
    assert gu["language"] == "gu" and "ચેતવણી" in gu["answer"]

def test_no_data_and_llm_used_only_with_context(cfg):
    db = DB()
    assert "No readings" in rag.answer("Is Alkapuri safe?", db, cfg)["answer"]
    seen = {}
    def fake(q, ctx): seen["ctx"] = ctx; return "ok"
    r = rag.answer("Is Alkapuri safe?", db, cfg, llm=fake)
    assert r["source"] == "llm" and [c["station"] for c in seen["ctx"]] == ["alkapuri"]
