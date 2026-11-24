# data_prep.py - generated from the notebook (Task 20). Edit the notebook, then regenerate.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

KEY_COLS = ['bedrooms',
 'bathrooms',
 'toilets',
 'parking_space',
 'title',
 'town',
 'state',
 'price']
RANDOM_STATE = 42
TEST_SIZE = 0.2
WRONG_LABEL = 'Anambara'
TOWN_TO_STATE = {'Lekki': 'Lagos',
 'Ibeju Lekki': 'Lagos',
 'Ajah': 'Lagos',
 'Alimosho': 'Lagos',
 'Ikoyi': 'Lagos',
 'Ifako-Ijaiye': 'Lagos',
 'Ikorodu': 'Lagos',
 'Ikeja': 'Lagos',
 'Victoria Island (VI)': 'Lagos',
 'Ketu': 'Lagos',
 'Ikotun': 'Lagos',
 'Gbagada': 'Lagos',
 'Agege': 'Lagos',
 'Ojo': 'Lagos',
 'Shomolu': 'Lagos',
 'Yaba': 'Lagos',
 'Lokogoma District': 'Abuja',
 'Guzape District': 'Abuja',
 'Mbora (Nbora)': 'Abuja',
 'Gwarinpa': 'Abuja',
 'Katampe': 'Abuja',
 'Jabi': 'Abuja',
 'Mowe Ofada': 'Ogun',
 'Mowe Town': 'Ogun',
 'Arepo': 'Ogun',
 'Magboro': 'Ogun',
 'Port Harcourt': 'Rivers',
 'Ibadan': 'Oyo',
 'Owerri Municipal': 'Imo',
 'Akure': 'Ondo'}
NO_TOUCH_TOWNS = {'Isheri North'}
ALWAYS_EXCLUDE_TOWNS = {'Karu'}
FALLBACK_MIN_ROWS = 5
FALLBACK_MIN_SHARE = 0.9
PRICE_MAX_RATIO = 10.0
MIN_GROUP = 10
REFERENCE_LEVELS = [['town', 'state', 'title'], ['state', 'title'], ['title']]

def clean_labels(df, town_to_state=TOWN_TO_STATE, no_touch=NO_TOUCH_TOWNS, always_exclude=ALWAYS_EXCLUDE_TOWNS,
                 wrong_label=WRONG_LABEL, fallback_min_rows=FALLBACK_MIN_ROWS,
                 fallback_min_share=FALLBACK_MIN_SHARE):
    """Reassign rows labelled with `wrong_label` to their real state. Returns a new frame."""
    out = df.copy()
    if "state_original" not in out.columns:
        out["state_original"] = out["state"]
    if "exclude_reason" not in out.columns:
        out["exclude_reason"] = ""
    out["label_fix"] = "unchanged"

    bad = out["state"].eq(wrong_label)
    town = out["town"]

    # Rule 1: reviewed town -> state table.
    planned = town.map(town_to_state)
    skip = town.isin(no_touch) | town.isin(always_exclude)   # towns this step must not reassign
    rule = bad & planned.notna() & ~skip
    out.loc[rule, "state"] = planned[rule]
    out.loc[rule, "label_fix"] = "mapped_rule"

    # Rule 2: majority state of the same town among correctly labelled rows.
    elsewhere = out.loc[~bad].groupby(["town", "state"]).size().rename("n").reset_index()
    elsewhere["total"] = elsewhere.groupby("town")["n"].transform("sum")
    elsewhere = (elsewhere.sort_values(["town", "n"], ascending=[True, False])
                 .drop_duplicates("town").set_index("town"))
    maj_state = town.map(elsewhere["state"])
    maj_share = town.map(elsewhere["n"] / elsewhere["total"])
    maj_total = town.map(elsewhere["total"])
    fallback = (bad & ~rule & ~skip & maj_state.notna()
                & (maj_share >= fallback_min_share) & (maj_total >= fallback_min_rows))
    out.loc[fallback, "state"] = maj_state[fallback]
    out.loc[fallback, "label_fix"] = "mapped_majority"

    # Anything still unresolved is kept in the table but excluded from modelling.
    unresolved = bad & ~rule & ~fallback
    out.loc[unresolved, "label_fix"] = "unresolved"
    out.loc[unresolved & out["exclude_reason"].eq(""), "exclude_reason"] = "unresolved_state"
    return out


def deduplicate(df, key_cols=KEY_COLS):
    '''Mark repeated listings (keep the first) and exclude them from modelling. Returns a new frame.'''
    out = df.copy()
    if "exclude_reason" not in out.columns:
        out["exclude_reason"] = ""
    eligible = out["exclude_reason"].eq("")
    is_dup = pd.Series(False, index=out.index)
    is_dup.loc[eligible] = out.loc[eligible].duplicated(subset=list(key_cols), keep="first")
    out["is_duplicate"] = is_dup
    out.loc[is_dup, "exclude_reason"] = "duplicate"
    return out


def split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    '''Random split of the eligible rows, stratified by region. Adds a `split` column ("train", "test" or "").'''
    out = df.copy()
    out["split"] = ""
    ids = out.index[out["exclude_reason"].eq("")]
    states = out.loc[ids, "state"]
    region = np.where(states.isin(["Lagos", "Abuja"]), states, "Other")
    train_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=random_state, stratify=region)
    out.loc[train_ids, "split"] = "train"
    out.loc[test_ids, "split"] = "test"
    return out


def build_price_reference(train_rows, levels=REFERENCE_LEVELS, min_group=MIN_GROUP):
    '''Median log10 price of each comparable-listing group, from the TRAINING rows only.'''
    tables = []
    for lvl in levels:
        g = (train_rows.assign(_lp=np.log10(train_rows["price"]))
             .groupby(lvl)["_lp"].agg(["median", "size"]).reset_index())
        g = g[g["size"] >= min_group].rename(columns={"median": "_ref"})[lvl + ["_ref"]]
        tables.append((lvl, g))
    return tables


def price_deviation(df, reference):
    '''log10(price) minus log10(reference median), using the most specific group available.'''
    ref = np.full(len(df), np.nan)
    used = np.array([""] * len(df), dtype=object)
    for lvl, g in reference:
        r = df[lvl].merge(g, on=lvl, how="left")["_ref"].to_numpy(dtype=float)
        take = np.isnan(ref) & ~np.isnan(r)
        ref[take] = r[take]
        used[take] = "+".join(lvl)
    return pd.DataFrame({"price_ref_median": 10 ** ref, "price_ref_level": used,
                         "price_dev_log10": np.log10(df["price"].to_numpy()) - ref}, index=df.index)


def flag_prices(df, reference, max_ratio=PRICE_MAX_RATIO):
    '''Flag eligible rows priced more than `max_ratio` times above or below their reference. Returns a new frame.'''
    out = df.copy()
    dev = price_deviation(out, reference)
    out = out.drop(columns=dev.columns, errors="ignore").join(dev)
    limit = np.log10(max_ratio)
    eligible = out["exclude_reason"].eq("")
    low = (out["price_dev_log10"] <= -limit) & eligible
    high = (out["price_dev_log10"] >= limit) & eligible
    out["price_flag"] = low | high
    out["price_flag_dir"] = np.where(low, "too_low", np.where(high, "too_high", ""))
    out.loc[out["price_flag"], "exclude_reason"] = "implausible_price"
    return out


def prepare_dataset(raw):
    '''Label correction, de-duplication, split and price rule on a raw frame. Returns the full audit table.'''
    out = raw.copy()
    out["state_original"] = out["state"]
    out["exclude_reason"] = ""
    out = clean_labels(out)
    out = deduplicate(out)
    out = split_data(out)
    reference = build_price_reference(out[out["split"] == "train"])
    return flag_prices(out, reference)


def model_frame(prepared, key_cols=KEY_COLS):
    '''The rows that go to modelling: kept rows, model columns plus the split (original row numbers preserved).'''
    kept = prepared.loc[prepared["exclude_reason"].eq(""), list(key_cols) + ["split"]]
    return kept

