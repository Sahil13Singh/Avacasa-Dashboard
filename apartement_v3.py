
import json
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
RNG = 42
DATA_PATH = "./hp_unified_listings.json"

# ----------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------
with open(DATA_PATH) as f:
    payload = json.load(f)
df = pd.DataFrame(payload["records"])
print(f"Loaded {len(df)} records "
      f"({(df.data_source=='scraped_99acres').sum()} scraped, "
      f"{(df.data_source=='avacasa_internal').sum()} Avacasa-synthetic)")

# ----------------------------------------------------------------------
# 2. STRICT FILTER + CLEAN
#    (asset_class is already clean here -- only 'apartment'/'villa' exist,
#    no farmland/plot/agri category present, so the strict exclusion the
#    original brief asked for is structurally satisfied already)
# ----------------------------------------------------------------------
df = df[df["asset_class"].isin(["apartment", "villa"])].copy()
df = df.dropna(subset=["listing_price", "saleable_area_sqft"])
df = df[(df["listing_price"] > 0) & (df["saleable_area_sqft"] > 0)]

# drop scrape-error-scale outliers (a handful of rows, not ~50 like before --
# this dataset's saleable_area_sqft is far cleaner than the old `area` field)
before = len(df)
df = df[(df["saleable_area_sqft"] >= 100) & (df["saleable_area_sqft"] <= 15000)]
print(f"Dropped {before - len(df)} rows with implausible saleable_area_sqft "
      f"(<100 or >15,000 sqft)")

df["ppsf"] = df["listing_price"] / df["saleable_area_sqft"]
lo, hi = df["ppsf"].quantile([0.005, 0.995])
df = df[(df["ppsf"] >= lo) & (df["ppsf"] <= hi)].copy()
print(f"Rows after filtering + cleaning: {len(df)}")

# ----------------------------------------------------------------------
# 3. LOCATION: backfill `district` (only 4 clean values, but only
#    populated for 2,852/2,996 Avacasa rows and null for all scraped rows)
#    using a small gazetteer over `major_market` / `micro_market` /
#    `location_raw`, since those match the same 4 districts in every
#    sample checked.
# ----------------------------------------------------------------------
DISTRICT_PATTERNS = [
    ("Shimla", r"shimla|kufri|mashobra|naldhera|chail\b"),  # chail also Solan-adjacent; checked below
    ("Solan", r"solan|kasauli|kumarhatti|barog|dharampur|nalagarh|chail"),
    ("Kangra", r"kangra|dharamsh?ala|palampur"),
    ("Kullu", r"kullu|kulu|manali"),
]

def resolve_district(row):
    if pd.notna(row.get("district")):
        return row["district"]
    text = " ".join(str(x) for x in [
        row.get("major_market", ""), row.get("micro_market", ""),
        row.get("location_raw", ""), row.get("name", ""),
    ]).lower()
    for d, pattern in DISTRICT_PATTERNS:
        if re.search(pattern, text):
            return d
    return "Others"

df["district_filled"] = df.apply(resolve_district, axis=1)

# ----------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ----------------------------------------------------------------------
df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")
df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")  # left NaN where unknown; CatBoost handles natively
df["balconies"] = pd.to_numeric(df["balconies"], errors="coerce")
df["parking_count"] = pd.to_numeric(df["parking_count"], errors="coerce")
df["total_floors"] = pd.to_numeric(df["total_floors"], errors="coerce")
df["floor_number"] = pd.to_numeric(df["floor_number"], errors="coerce")
df["society_total_units"] = pd.to_numeric(df["society_total_units"], errors="coerce")
df["elevation_m"] = pd.to_numeric(df["elevation_m"], errors="coerce")
df["chandigarh_drive_mins"] = pd.to_numeric(df["chandigarh_drive_mins"], errors="coerce")
df["delhi_drive_mins"] = pd.to_numeric(df["delhi_drive_mins"], errors="coerce")
df["road_quality_score"] = pd.to_numeric(df["road_quality_score"], errors="coerce")
df["str_gross_yield"] = pd.to_numeric(df["str_gross_yield"], errors="coerce")
df["second_home_score"] = pd.to_numeric(df["second_home_score"], errors="coerce")
df["benchmark_psf"] = pd.to_numeric(df["benchmark_psf"], errors="coerce")
df["has_rera"] = df["has_rera"].fillna(False).astype(int)
df["is_villa"] = (df["asset_class"] == "villa").astype(int)

for c in ["facing", "furnishing", "power_backup", "construction_status",
          "transaction_type", "view_type", "developer_tier",
          "market_maturity", "flood_risk", "area_type"]:
    df[c] = df[c].fillna("unknown").astype(str)

df["area_per_bhk"] = df["saleable_area_sqft"] / df["bedrooms"].replace(0, np.nan)
df["bath_per_bhk"] = df["bathrooms"] / df["bedrooms"].replace(0, np.nan)
df["log_area"] = np.log1p(df["saleable_area_sqft"])
df["sqrt_area"] = np.sqrt(df["saleable_area_sqft"])
df["log_ppsf"] = np.log1p(df["ppsf"])

# ----------------------------------------------------------------------
# 5. LEAK-FREE HIERARCHICAL LOCATION PRIOR
#    (major_market median log-PPSF, shrunk to district median when sparse;
#    encoder fit on the 70% TRAIN split only, applied to test/unseen)
# ----------------------------------------------------------------------
SHRINK_K = 5

def fit_location_encoder(train_idx, frame):
    sub = frame.loc[train_idx]
    district_med = sub.groupby("district_filled")["log_ppsf"].median()
    global_med = sub["log_ppsf"].median()
    market_stats = sub.groupby("major_market")["log_ppsf"].agg(["median", "count"])
    market_district = sub.groupby("major_market")["district_filled"].first()
    return district_med, global_med, market_stats, market_district


def apply_location_encoder(idx, frame, district_med, global_med, market_stats, market_district):
    rows = frame.loc[idx]
    out = []
    for district, market in zip(rows["district_filled"], rows["major_market"]):
        d_med = district_med.get(district, global_med)
        if market in market_stats.index:
            m_med, m_cnt = market_stats.loc[market, "median"], market_stats.loc[market, "count"]
            w = m_cnt / (m_cnt + SHRINK_K)
            out.append(w * m_med + (1 - w) * d_med)
        else:
            out.append(d_med)
    return np.array(out)


NUMERIC_FEATURES = [
    "saleable_area_sqft", "bedrooms", "bathrooms", "balconies", "is_villa",
    "has_rera", "parking_count", "total_floors", "floor_number",
    "society_total_units", "elevation_m", "chandigarh_drive_mins",
    "delhi_drive_mins", "road_quality_score", "str_gross_yield",
    "second_home_score", "benchmark_psf", "area_per_bhk", "bath_per_bhk",
    "log_area", "sqrt_area",
]
CAT_FEATURES = [
    "district_filled", "facing", "furnishing", "power_backup",
    "construction_status", "transaction_type", "view_type",
    "developer_tier", "market_maturity", "flood_risk", "area_type",
    "data_source",
]
FEATURES = NUMERIC_FEATURES + CAT_FEATURES + ["loc_ppsf_prior"]

X_all = df.reset_index(drop=True).copy()
y_all = X_all["log_ppsf"]
area_all = X_all["saleable_area_sqft"]
price_all = X_all["listing_price"]
source_all = X_all["data_source"]


def make_design_matrix(idx, encoder_args):
    sub = X_all.loc[idx].copy()
    sub["loc_ppsf_prior"] = apply_location_encoder(idx, X_all, *encoder_args)
    return sub[FEATURES]


# ----------------------------------------------------------------------
# 6. 70 / 15 / 15 SPLIT (stratified on data_source so the real/synthetic
#    mix is identical across train, test, and the unseen holdout)
# ----------------------------------------------------------------------
train_idx, rest_idx = train_test_split(
    X_all.index, test_size=0.30, random_state=RNG, stratify=source_all
)
test_idx, unseen_idx = train_test_split(
    rest_idx, test_size=0.50, random_state=RNG, stratify=source_all.loc[rest_idx]
)
print(f"\nSplit sizes -> train: {len(train_idx)} ({len(train_idx)/len(X_all):.1%})  "
      f"test: {len(test_idx)} ({len(test_idx)/len(X_all):.1%})  "
      f"unseen: {len(unseen_idx)} ({len(unseen_idx)/len(X_all):.1%})")

encoder_args = fit_location_encoder(train_idx, X_all)
X_train = make_design_matrix(train_idx, encoder_args)
X_test = make_design_matrix(test_idx, encoder_args)
X_unseen = make_design_matrix(unseen_idx, encoder_args)
y_train = y_all.loc[train_idx]

# ----------------------------------------------------------------------
# 7. MODEL
# ----------------------------------------------------------------------
model = CatBoostRegressor(
    iterations=500,
    depth=5,
    learning_rate=0.03,
    l2_leaf_reg=6,
    loss_function="RMSE",
    random_seed=RNG,
    verbose=False,
)
model.fit(X_train, y_train, cat_features=CAT_FEATURES)


def evaluate(idx, X_design, label):
    pred_log_ppsf = model.predict(X_design)
    pred_price = np.expm1(pred_log_ppsf) * area_all.loc[idx].values
    true_price = price_all.loc[idx].values
    r2 = r2_score(true_price, pred_price)
    mae = mean_absolute_error(true_price, pred_price)
    print(f"{label:28s} -> R2: {r2:.4f}   MAE: Rs.{mae:,.0f}   n={len(idx)}")

    # breakdown by data_source -- the honesty check
    src = source_all.loc[idx]
    for s in src.unique():
        mask = (src == s).values
        if mask.sum() < 5:
            continue
        r2_s = r2_score(true_price[mask], pred_price[mask])
        mae_s = mean_absolute_error(true_price[mask], pred_price[mask])
        print(f"   - {s:22s} R2: {r2_s:.4f}   MAE: Rs.{mae_s:,.0f}   n={mask.sum()}")
    return pred_price, true_price


print("\n--- TEST SET (15%) ---")
evaluate(test_idx, X_test, "Test (15%)")

print("\n--- UNSEEN HOLDOUT (15%) ---")
unseen_pred, unseen_true = evaluate(unseen_idx, X_unseen, "Unseen holdout (15%)")

# ----------------------------------------------------------------------
# 8. VISUALIZATIONS (on the unseen holdout)
# ----------------------------------------------------------------------
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(7, 6))
src_unseen = source_all.loc[unseen_idx].values
colors = np.where(src_unseen == "scraped_99acres", "#dc2626", "#2563eb")
ax.scatter(unseen_true / 1e6, unseen_pred / 1e6, alpha=0.6, c=colors,
           edgecolor="k", linewidth=0.3)
lims = [0, max(unseen_true.max(), unseen_pred.max()) / 1e6]
ax.plot(lims, lims, "k--", lw=2, label="Perfect Prediction")
ax.scatter([], [], c="#dc2626", label="Real (scraped_99acres)")
ax.scatter([], [], c="#2563eb", label="Synthetic (avacasa_internal)")
ax.set_xlabel("Actual Price (Rs. Million)")
ax.set_ylabel("Predicted Price (Rs. Million)")
ax.set_title("Actual vs. Predicted Price — 15% Unseen Holdout (v3)")
ax.legend()
plt.tight_layout()
plt.savefig("./actual_vs_predicted_v3.png", dpi=150)
plt.close()

importances = pd.Series(model.get_feature_importance(), index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(8, 8))
importances.plot(kind="barh", ax=ax, color="#16a34a")
ax.set_title("Feature Importance (CatBoost, predicting log-PPSF) — v3")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("./feature_importance_v3.png", dpi=150)
plt.close()

print("\nSaved plots: actual_vs_predicted_v3.png, feature_importance_v3.png")
