"""Feature engineering and model building blocks shared by the training notebook and the Streamlit app.

The saved model pickles references to functions and classes in this module, so the module must be
importable (kept next to the app) both when training and when serving.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

# Columns a user (or the app) must supply.
NUM_INPUTS = ["bedrooms", "bathrooms", "toilets", "parking_space"]
CAT_INPUTS = ["title", "town", "state"]
RAW_COLS = NUM_INPUTS + CAT_INPUTS

# Columns the model actually sees.
ENGINEERED = ["rooms_total", "bath_per_bed", "toilet_per_bed"]
MODEL_NUM_COLS = NUM_INPUTS + ENGINEERED
MODEL_CAT_COLS = ["title", "state", "town_state"]

MIN_CAT_COUNT = 5   # categories seen fewer times than this are pooled into one "infrequent" level


def region_of(state):
    """Group states into the three regions used for stratification and error analysis."""
    return state if state in ("Lagos", "Abuja") else "Other"


def region_series(states):
    return pd.Series(states).map(region_of).to_numpy()


def add_features(X):
    """Raw listing columns -> model columns. Works on a DataFrame with at least RAW_COLS."""
    X = pd.DataFrame(X).copy()
    for c in NUM_INPUTS:
        X[c] = X[c].astype(float)
    X["town_state"] = X["town"].astype(str) + " | " + X["state"].astype(str)
    X["rooms_total"] = X["bedrooms"] + X["bathrooms"] + X["toilets"]
    X["bath_per_bed"] = X["bathrooms"] / X["bedrooms"].clip(lower=1)
    X["toilet_per_bed"] = X["toilets"] / X["bedrooms"].clip(lower=1)
    return X[MODEL_NUM_COLS + MODEL_CAT_COLS]


class GroupMedianBaseline(RegressorMixin, BaseEstimator):
    """Simple lookup baseline: median log price of comparable training listings.

    Tries town+type, then state+type, then type alone; a group needs `min_group` listings to be used.
    """

    LEVELS = (("town_state", "title"), ("state", "title"), ("title",))

    def __init__(self, min_group=5):
        self.min_group = min_group

    def fit(self, X, y):
        d = pd.DataFrame(X).copy()
        d["_y"] = np.asarray(y, dtype=float)
        self.tables_ = []
        for level in self.LEVELS:
            g = d.groupby(list(level))["_y"].agg(["median", "size"])
            g = g[g["size"] >= self.min_group]
            self.tables_.append(g["median"].to_dict())
        self.global_ = float(d["_y"].median())
        return self

    def predict(self, X):
        d = pd.DataFrame(X)
        out = np.full(len(d), np.nan)
        for level, table in zip(self.LEVELS, self.tables_):
            keys = list(zip(*[d[c] for c in level])) if len(level) > 1 else list(d[level[0]])
            vals = np.array([table.get(k, np.nan) for k in keys], dtype=float)
            take = np.isnan(out) & ~np.isnan(vals)
            out[take] = vals[take]
        out[np.isnan(out)] = self.global_
        return out


def _preprocessor(kind):
    if kind == "onehot_scaled":      # linear models
        num = StandardScaler()
    else:
        num = "passthrough"
    if kind == "native":             # tree models with native categorical support
        cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                             encoded_missing_value=np.nan, min_frequency=MIN_CAT_COUNT)
    else:
        cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_CAT_COUNT,
                            sparse_output=False)
    return ColumnTransformer([("num", num, MODEL_NUM_COLS), ("cat", cat, MODEL_CAT_COLS)])


def native_categorical_mask():
    """Mask for HistGradientBoostingRegressor(categorical_features=...) after the 'native' preprocessor."""
    return [False] * len(MODEL_NUM_COLS) + [True] * len(MODEL_CAT_COLS)


def build_pipeline(model, kind="onehot"):
    """features -> preprocessing -> model. `kind` is 'onehot', 'onehot_scaled' or 'native'.
    Input is a DataFrame with RAW_COLS; the model predicts ln(price)."""
    steps = [("features", FunctionTransformer(add_features))]
    if kind != "none":
        steps.append(("prep", _preprocessor(kind)))
    steps.append(("model", model))
    return Pipeline(steps)
