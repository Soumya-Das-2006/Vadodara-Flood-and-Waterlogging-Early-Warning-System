from unittest import mock
import pathlib
from streamlit.testing.v1 import AppTest

APP = str(pathlib.Path(__file__).resolve().parents[1] / "dashboard" / "app.py")

STATIONS = [{"id": "akota", "name": "Akota Bridge", "name_hi": "अकोटा ब्रिज", "name_gu": "અકોટા બ્રિજ", "lat": 22.3, "lon": 73.16,
             "danger_cm": 80, "alt_route": {}, "status": "warning", "current_cm": 61.0, "flood_probability": 0.8,
             "eta_minutes": 120, "as_of": "2026-07-01T10:00:00+05:30"}]

def _fake(method, url, **k):
    r = mock.Mock(); r.raise_for_status = lambda: None
    r.json = lambda: STATIONS if url.endswith("/stations") else []
    return r

def test_dashboard_renders_and_switches_language():
    with mock.patch("requests.request", _fake):
        at = AppTest.from_file(APP, default_timeout=30).run()
        assert not at.exception
        at.sidebar.selectbox[0].set_value("gu").run()
        assert not at.exception and any("વડોદરા" in t.value for t in at.title)

def test_dashboard_handles_api_down():
    def down(*a, **k): raise ConnectionError("x")
    with mock.patch("requests.request", down):
        at = AppTest.from_file(APP, default_timeout=30).run()
        assert not at.exception and at.error
