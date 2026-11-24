"""Tests for the prediction code the Streamlit app relies on. Run with:  python -m pytest -q

These tests train a tiny model on made-up data, so they run in seconds and do not need the real dataset.
If models/model.joblib exists, the last test also checks the real saved model.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import house_features as hf  # noqa: E402
import predictor  # noqa: E402


@pytest.fixture(scope="module")
def bundle():
    rng = np.random.default_rng(0)
    towns = [("Lagos", "Lekki", 1.0), ("Lagos", "Ajah", 0.6), ("Abuja", "Gwarinpa", 0.8), ("Oyo", "Ibadan", 0.3)]
    rows = []
    for _ in range(600):
        state, town, f = towns[rng.integers(len(towns))]
        bed = int(rng.integers(2, 7))
        rows.append({"bedrooms": bed, "bathrooms": bed + 1, "toilets": bed + 1, "parking_space": int(rng.integers(1, 6)),
                     "title": str(rng.choice(["Detached Duplex", "Block of Flats"])), "town": town, "state": state,
                     "price": 2e7 * f * bed * float(np.exp(rng.normal(0, 0.3)))})
    df = pd.DataFrame(rows)
    model = hf.build_pipeline(Ridge(alpha=1.0), kind="onehot_scaled").fit(df[hf.RAW_COLS], np.log(df["price"]))
    meta = {"options": predictor.build_options(df)}
    return {"model": model, "meta": meta}


GOOD = dict(state="Lagos", town="Lekki", title="Detached Duplex", bedrooms=4, bathrooms=5, toilets=5, parking_space=3)


def test_prediction_is_a_positive_number(bundle):
    p = predictor.predict(bundle, **GOOD)
    assert math.isfinite(p.estimate) and p.estimate > 0
    assert p.region == "Lagos"


def test_other_region_gets_a_reliability_note(bundle):
    p = predictor.predict(bundle, **{**GOOD, "state": "Oyo", "town": "Ibadan"})
    assert p.region == "Other"
    assert any("Lagos and Abuja" in n for n in p.notes)


def test_more_bedrooms_costs_more_in_this_toy_model(bundle):
    small = predictor.predict(bundle, **{**GOOD, "bedrooms": 2, "bathrooms": 3, "toilets": 3}).estimate
    large = predictor.predict(bundle, **{**GOOD, "bedrooms": 6, "bathrooms": 7, "toilets": 7}).estimate
    assert large > small


@pytest.mark.parametrize("bad", [
    {"town": "Nowhere"},                      # unknown town
    {"state": "Abuja"},                       # valid town, wrong state
    {"title": "Castle"},                      # unknown property type
    {"bedrooms": 40},                         # outside the trained range
    {"parking_space": 0},                     # below the trained range
])
def test_invalid_inputs_are_rejected(bundle, bad):
    with pytest.raises(ValueError):
        predictor.predict(bundle, **{**GOOD, **bad})


def test_unseen_town_never_reaches_the_model(bundle):
    # The model can handle unseen categories, but the app only offers known towns.
    assert "Nowhere" not in bundle["meta"]["options"]["locations"]["Lagos"]


def test_format_naira():
    assert predictor.format_naira(85_000_000) == "₦85.0M"
    assert predictor.format_naira(1_250_000_000) == "₦1.25B"
    assert predictor.format_naira(660_000) == "₦660K"
    assert predictor.format_naira(500) == "₦500"


def test_group_median_baseline_falls_back_to_broader_groups():
    X = pd.DataFrame({"town_state": ["A | X"] * 6 + ["B | X"] * 2, "state": ["X"] * 8, "title": ["T"] * 8})
    y = np.array([1.0] * 6 + [5.0, 5.0])
    m = hf.GroupMedianBaseline(min_group=5).fit(X, y)
    new = pd.DataFrame({"town_state": ["A | X", "B | X", "C | Y"], "state": ["X", "X", "Y"], "title": ["T"] * 3})
    pred = m.predict(new)
    assert pred[0] == 1.0                      # town + type group is large enough
    assert pred[1] == 1.0                      # town too small -> state + type median (1.0)
    assert pred[2] == 1.0                      # unseen state -> type median


@pytest.mark.skipif(not (Path("models/model.joblib").exists() and Path("models/model_meta.json").exists()),
                    reason="saved model not found")
def test_saved_model_loads_and_predicts():
    bundle = predictor.load_bundle("models")
    meta = bundle["meta"]
    state = next(iter(meta["options"]["locations"]))
    town = meta["options"]["locations"][state][0]
    ranges = meta["options"]["ranges"]
    p = predictor.predict(bundle, state=state, town=town, title=meta["options"]["titles"][0],
                          bedrooms=ranges["bedrooms"][0], bathrooms=ranges["bathrooms"][0],
                          toilets=ranges["toilets"][0], parking_space=ranges["parking_space"][0])
    assert p.estimate > 0
