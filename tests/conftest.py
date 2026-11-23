import pathlib, sys
import pytest
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, SEED
from src.synth import synth_rain, make_context
from src.build_dataset import make_dataset


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def tiny_df(cfg):
    """~7 years of SYNTHETIC data (test fixture only)."""
    rain = synth_rain("2019-01-01", "2025-12-31 23:00", SEED)
    data, _, _ = make_dataset(rain, make_context(rain), cfg, verbose=False)
    return data


@pytest.fixture(scope="session")
def tiny_model(tiny_df, tmp_path_factory):
    from src.train import run
    out = tmp_path_factory.mktemp("models")
    run(out_dir=out, quick=True, skip_gru=True, figs=False, df=tiny_df)
    return out / "flood_model.joblib"


@pytest.fixture(scope="session")
def predictor(tiny_model, cfg):
    from src.predict import FloodPredictor
    return FloodPredictor(tiny_model, cfg)
