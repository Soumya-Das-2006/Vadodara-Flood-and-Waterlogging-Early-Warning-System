from src.level_sim import StationSim, soil_proxy
from src.config import load_config

def test_level_rises_with_rain_and_recedes():
    sim = StationSim.from_cfg(load_config()["stations"]["akota"]["sim"])
    dry = sim.level
    peak = max(sim.step(30.0, 1.0) for _ in range(8))
    assert peak > dry + 50
    after = [sim.step(0.0, 1.0) for _ in range(30)]
    assert after[-1] < peak and after[-1] > 0

def test_wetter_soil_means_more_runoff():
    cfg = load_config()["stations"]["alkapuri"]["sim"]
    a, b = StationSim.from_cfg(cfg), StationSim.from_cfg(cfg)
    for _ in range(48): b.step(5.0, 1.0)
    for s in (a, b): s.Lf = 0.0; s.R = 0.0; s.m24 = 0.0
    assert b.step(20.0, 1.0) > a.step(20.0, 1.0)

def test_soil_proxy_bounds():
    sm = soil_proxy([100.0] * 100 + [0.0] * 500)
    assert 0.05 <= min(sm) and max(sm) <= 0.45
