from backend.app.telegram import TelegramNotifier

T = {"en": "a", "hi": "b", "gu": "c"}

def _n(clock, **k):
    sent = []
    return TelegramNotifier(token="x", chat_id="y", clock=lambda: clock[0], sender=sent.append, **k), sent

def test_not_configured_is_skipped():
    assert TelegramNotifier(token="", chat_id="").send("akota", "warning", T) == "not_configured"

def test_dedupe_and_rate_limit():
    clk = [1000.0]; n, sent = _n(clk, min_interval=600)
    assert n.send("akota", "warning", T) == "sent"
    assert n.send("akota", "warning", T) == "duplicate"
    assert n.send("akota", "safe", {"en": "z", "hi": "b", "gu": "c"}) == "rate_limited"
    assert n.send("alkapuri", "warning", T) == "sent"                    # other station unaffected
    clk[0] += 700
    assert n.send("akota", "safe", {"en": "z", "hi": "b", "gu": "c"}) == "sent"
    assert len(sent) == 3

def test_danger_bypasses_interval_and_message_is_trilingual():
    clk = [0.0]; n, sent = _n(clk, min_interval=600)
    n.send("akota", "warning", T)
    assert n.send("akota", "danger", {"en": "D", "hi": "h", "gu": "g"}) == "sent"
    assert "D" in sent[-1] and "h" in sent[-1] and "g" in sent[-1]

def test_global_cap_and_sender_error():
    clk = [0.0]; n, _ = _n(clk, max_per_minute=2, min_interval=0)
    assert [n.send(f"s{i}", "warning", {"en": str(i), "hi": "", "gu": ""}) for i in range(3)] == ["sent", "sent", "rate_limited"]
    bad = TelegramNotifier(token="x", chat_id="y", sender=lambda m: (_ for _ in ()).throw(RuntimeError("x")))
    assert bad.send("a", "warning", T).startswith("error")
