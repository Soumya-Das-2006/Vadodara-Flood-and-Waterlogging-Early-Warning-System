"""Shared constants and config loading."""
import os
import pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 42
TZ = "Asia/Kolkata"
HORIZONS = [1, 2, 3]            # forecast horizons in hours
DATA_RAW = ROOT / "data" / "raw"
DATA_SYN = ROOT / "data" / "synthetic"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"
FIGS = DOCS / "figures"


def load_config(path=None):
    path = pathlib.Path(path or os.environ.get("CONFIG_PATH", ROOT / "config" / "stations.yaml"))
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def station_ids(cfg=None):
    cfg = cfg or load_config()
    return list(cfg["stations"])
