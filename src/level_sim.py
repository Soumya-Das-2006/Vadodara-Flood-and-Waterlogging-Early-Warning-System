"""Physically based SYNTHETIC water-level model (rain -> runoff -> drain/river level).

Used (a) to build the SYNTHETIC-TARGET level series from rainfall and
(b) by the sensor simulator. Same equations in both, so the simulator
behaves like the training data.

  soil:    sm   <- sm*exp(-dt/60) + 0.006*rain*dt       (antecedent wetness proxy)
  runoff:  rain*(0.35 + 0.65*sm_norm)
  street:  Lf   <- Lf + (c*runoff - Lf) * (1-(1-k)^dt)   (fast drain/street response)
  river:   m24  <- EMA of rain (tau 24 h);  R <- R + (cr*m24 - R)*(1-(1-kr)^dt)  (slow)
  level  = base + Lf + R
"""
import math

SM_MIN, SM_MAX = 0.05, 0.45
SM_DECAY_H = 60.0
SM_GAIN = 0.006
SM_START = 0.12


def soil_step(sm, rain, dt=1.0):
    sm = sm * math.exp(-dt / SM_DECAY_H) + SM_GAIN * rain * dt
    return min(SM_MAX, max(SM_MIN, sm))


def soil_proxy(rain_values, sm0=SM_START):
    """Vector version: soil moisture proxy from an hourly rain array (DERIVED)."""
    out, sm = [], sm0
    for r in rain_values:
        sm = soil_step(sm, 0.0 if r != r else r)   # NaN rain counts as 0
        out.append(sm)
    return out


def discharge_proxy(rain_values, lag_h=24, tau_h=72.0):
    """Upstream discharge index (DERIVED, arbitrary units): linear reservoir, lagged."""
    out, q = [], 0.0
    a = math.exp(-1.0 / tau_h)
    for r in rain_values:
        q = q * a + (0.0 if r != r else r) * (1 - a)
        out.append(q)
    return [float("nan")] * lag_h + out[:-lag_h] if lag_h else out


class StationSim:
    def __init__(self, c, k, cr, kr, base=5.0, sm=SM_START):
        self.c, self.k, self.cr, self.kr, self.base = c, k, cr, kr, base
        self.sm, self.Lf, self.R, self.m24 = sm, 0.0, 0.0, 0.0

    def step(self, rain, dt=1.0):
        """Advance dt hours with constant rain (mm/h). Returns noise-free level (cm)."""
        self.sm = soil_step(self.sm, rain, dt)
        sm_norm = (self.sm - SM_MIN) / (SM_MAX - SM_MIN)
        runoff = rain * (0.35 + 0.65 * sm_norm)
        self.Lf += (self.c * runoff - self.Lf) * (1 - (1 - self.k) ** dt)
        self.m24 += (rain - self.m24) * (1 - math.exp(-dt / 24.0))
        self.R += (self.cr * self.m24 - self.R) * (1 - (1 - self.kr) ** dt)
        return self.level

    @property
    def level(self):
        return self.base + self.Lf + self.R

    @classmethod
    def from_cfg(cls, sim_cfg):
        return cls(sim_cfg["c"], sim_cfg["k"], sim_cfg["cr"], sim_cfg["kr"])
