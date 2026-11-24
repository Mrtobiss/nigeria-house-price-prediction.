"""Load the saved model and turn user inputs into a price estimate.

This module has no Streamlit dependency, so it is easy to test. The Streamlit app (app.py) only handles the
interface and calls `predict`.
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

import house_features  # noqa: F401  (must be importable so joblib can unpickle the model)
from house_features import RAW_COLS, region_of

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
THIN_TOWN_ROWS = 10        # towns with fewer training listings get a warning


def format_naira(x):
    """85_000_000 -> '₦85.0M'; 1_250_000_000 -> '₦1.25B'."""
    x = float(x)
    for limit, suffix, nd in [(1e12, "T", 2), (1e9, "B", 2), (1e6, "M", 1), (1e3, "K", 0)]:
        if abs(x) >= limit:
            return f"₦{x / limit:,.{nd}f}{suffix}"
    return f"₦{x:,.0f}"


def build_options(train):
    """Everything the app needs to know about the training data: valid inputs, ranges and data volume per town."""
    locations = (train[["state", "town"]].drop_duplicates().sort_values(["state", "town"]))
    return {
        "titles": sorted(train["title"].unique().tolist()),
        "locations": {s: sorted(g["town"].tolist()) for s, g in locations.groupby("state")},
        "ranges": {c: [int(train[c].min()), int(train[c].max())]
                   for c in ["bedrooms", "bathrooms", "toilets", "parking_space"]},
        "town_counts": (train["town"] + " | " + train["state"]).value_counts().astype(int).to_dict(),
    }


def save_bundle(model, meta, model_dir=DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib", compress=3)
    (model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_bundle(model_dir=DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    model_path, meta_path = model_dir / "model.joblib", model_dir / "model_meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Model files not found in {model_dir}. Run the notebook (Task 29) to create model.joblib and model_meta.json.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {"model": joblib.load(model_path), "meta": meta}


@dataclass
class Prediction:
    estimate: float
    region: str
    town_listings: int
    notes: list = field(default_factory=list)


def predict(bundle, *, state, town, title, bedrooms, bathrooms, toilets, parking_space):
    """Estimate the asking price. Raises ValueError for inputs the model was never trained on."""
    meta, model = bundle["meta"], bundle["model"]

    if title not in meta["options"]["titles"]:
        raise ValueError(f"Unknown property type: {title!r}.")
    if town not in meta["options"]["locations"].get(state, []):
        raise ValueError(f"{town!r} is not a known town in {state!r}.")
    values = {"bedrooms": bedrooms, "bathrooms": bathrooms, "toilets": toilets, "parking_space": parking_space}
    for name, v in values.items():
        lo, hi = meta["options"]["ranges"][name]
        if not (lo <= v <= hi):
            raise ValueError(f"{name} must be between {lo} and {hi} (got {v}).")

    row = pd.DataFrame([{**values, "title": title, "town": town, "state": state}])[RAW_COLS]
    ln_price = float(model.predict(row)[0])

    region = region_of(state)
    n_town = int(meta["options"]["town_counts"].get(f"{town} | {state}", 0))

    notes = []
    if n_town < THIN_TOWN_ROWS:
        notes.append(f"Only {n_town} listings in {town} were used for training, so this estimate is less reliable.")
    if region == "Other":
        notes.append("Most training listings are in Lagos and Abuja; estimates elsewhere are less reliable.")

    return Prediction(
        estimate=math.exp(ln_price),
        region=region, town_listings=n_town, notes=notes,
    )
