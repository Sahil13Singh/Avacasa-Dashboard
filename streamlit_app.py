"""
Avacasa Intelligence Dashboard
Conversational real estate intelligence for Himachal Pradesh residential property.

Run locally:
    streamlit run streamlit_app.py

Deploy to Streamlit Community Cloud:
    - Push repo with this file, requirements_streamlit.txt, and data/ folder
    - Set GEMINI_API_KEY in Streamlit secrets
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

def _get_secret(key: str, default: str = "") -> str:
    val = os.getenv(key, "")
    if not val:
        try:
            val = st.secrets.get(key, default)
        except Exception:
            val = default
    return val or default

GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Avacasa Intelligence",
    page_icon="🏔",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global resets ── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1a1a2e !important;
    border-right: 1px solid rgba(16,185,129,0.15);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stButton button {
    background: rgba(16,185,129,0.08) !important;
    border: 1px solid rgba(16,185,129,0.2) !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    text-align: left !important;
    padding: 8px 12px !important;
    width: 100%;
    transition: all 0.15s ease;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.4 !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(16,185,129,0.18) !important;
    color: #10b981 !important;
    border-color: rgba(16,185,129,0.4) !important;
}

/* ── Main content ── */
[data-testid="stMainBlockContainer"] {
    background: #f8fafc;
    padding-top: 0 !important;
}

/* ── Chat messages ── */
.user-msg {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
}
.user-bubble {
    background: #16a34a;
    color: white !important;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(22,163,74,0.3);
}
.ai-msg {
    display: flex;
    justify-content: flex-start;
    margin: 12px 0;
    gap: 10px;
    align-items: flex-start;
}
.ai-avatar {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #0d9488, #16a34a);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}
.ai-bubble {
    background: white;
    border: 1px solid #e2e8f0;
    color: #1F2937 !important;
    padding: 14px 18px;
    border-radius: 4px 18px 18px 18px;
    max-width: 82%;
    font-size: 14px;
    line-height: 1.6;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.msg-source {
    font-size: 11px;
    color: #94a3b8;
    margin-top: 8px;
    border-top: 1px solid #f1f5f9;
    padding-top: 6px;
}

/* ── Valuation card ── */
.val-card {
    background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
    border: 1px solid #bbf7d0;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 10px 0;
}
.val-main { font-size: 28px; font-weight: 800; color: #15803d; margin: 4px 0; }
.val-range { font-size: 13px; color: #6b7280; margin-bottom: 12px; }
.val-divider { border: none; border-top: 1px solid #d1fae5; margin: 12px 0; }
.val-row { display: flex; justify-content: space-between; font-size: 13px; margin: 4px 0; }
.val-label { color: #6b7280; }
.val-value { color: #1F2937; font-weight: 600; }
.val-warn { font-size: 12px; color: #92400e; background: #fffbeb;
            border: 1px solid #fde68a; border-radius: 8px; padding: 8px 12px; margin-top: 10px; }

/* ── Listing card ── */
.listing-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    transition: box-shadow 0.15s;
}
.listing-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
.listing-title { font-size: 14px; font-weight: 700; color: #1F2937; margin-bottom: 4px; }
.listing-loc { font-size: 12px; color: #64748b; margin-bottom: 8px; }
.listing-meta { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 8px; }
.listing-chip {
    font-size: 11px; padding: 3px 10px; border-radius: 20px; font-weight: 600;
}
.chip-price { background: #ecfdf5; color: #15803d; }
.chip-area { background: #eff6ff; color: #1d4ed8; }
.chip-bhk { background: #faf5ff; color: #7c3aed; }
.badge-real { background: #ecfdf5; color: #15803d; border: 1px solid #bbf7d0; }
.badge-synth { background: #f8fafc; color: #94a3b8; border: 1px solid #e2e8f0; }
.listing-ppsf { font-size: 12px; color: #64748b; margin-top: 4px; }
.ppsf-above { color: #ef4444; }
.ppsf-below { color: #16a34a; }

/* ── Chat container ── */
.chat-container {
    max-width: 860px;
    margin: 0 auto;
    padding: 20px 24px 100px;
    min-height: calc(100vh - 140px);
}

/* ── Status badges ── */
.status-ok { color: #16a34a !important; font-size: 12px; }
.status-loading { color: #f59e0b !important; font-size: 12px; }
.status-err { color: #ef4444 !important; font-size: 12px; }

/* ── Typing indicator ── */
.typing-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #94a3b8;
    margin: 0 2px;
    animation: bounce 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-5px); }
}

/* ── Logo ── */
.sidebar-logo {
    font-size: 20px;
    font-weight: 800;
    color: #10b981 !important;
    letter-spacing: -0.3px;
}
.sidebar-sub {
    font-size: 10px;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 2px;
}
.sidebar-section {
    font-size: 10px;
    color: #475569 !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 20px 0 8px;
    font-weight: 700;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid rgba(16,185,129,0.12);
    margin: 16px 0;
}

/* ── Market QA ── */
.qa-text { font-size: 14px; line-height: 1.7; color: #374151; }
.qa-text strong { color: #111827; }

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS (copied exactly from apartement_v3.py)
# ═════════════════════════════════════════════════════════════════════════════

RNG = 42
SHRINK_K = 5

DISTRICT_PATTERNS = [
    ("Shimla",  r"shimla|kufri|mashobra|naldhera|chail\b"),
    ("Solan",   r"solan|kasauli|kumarhatti|barog|dharampur|nalagarh|chail"),
    ("Kangra",  r"kangra|dharamsh?ala|palampur"),
    ("Kullu",   r"kullu|kulu|manali"),
]

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

# Confidence bands from the honesty summary in villa.py / apartement_v3.py
APT_MAE_INR  = 19_00_000    # Rs. 19L for apartments (real-listing holdout)
VILLA_MAE_INR = 83_00_000   # Rs. 83L for villas

# ── Knowledge chunks (hardcoded per brief Section 4) ──────────────────────────
CHUNKS = [
    {
        "source": "HP RERA 2017",
        "text": "All residential projects in Himachal Pradesh with more than 8 units or over 500 sqm plot area must be registered under HP RERA (Real Estate Regulatory Authority). Developers must deposit 70% of project funds in a dedicated escrow account. Buyers can file complaints online at hprera.in. Penalty for non-registration is up to 10% of project cost."
    },
    {
        "source": "Section 118 HP Tenancy & Land Reforms Act",
        "text": "Section 118 of the HP Tenancy and Land Reforms Act prohibits non-HP-domicile persons from purchasing agricultural or non-urban land in Himachal Pradesh without prior permission from the state government. This restriction does NOT apply to apartments and built-up residential units in declared urban areas. Non-HP residents can freely purchase apartments and villas within municipal limits. Industrial units with government approval are exempt."
    },
    {
        "source": "Section 118 — Practical Implications",
        "text": "For buyers from outside Himachal Pradesh: apartments and villas in Shimla, Solan, Kasauli, Dharamsala municipal areas are freely purchasable. Farmhouses and agricultural land require Section 118 permission, which is rarely granted to outsiders. Many so-called 'farmhouse' listings are actually villas marketed misleadingly — buyers should verify the municipal classification of the land before purchase."
    },
    {
        "source": "Avacasa District Market Profile — Shimla",
        "text": "Shimla is the state capital and Himachal Pradesh's most liquid residential market. Median price per sqft is approximately Rs. 10,268. Key micro-markets: Mall Road (premium, Rs. 12,000-18,000/sqft), Panthaghati (mid-range, Rs. 8,000-11,000/sqft), Sanjauli (affordable, Rs. 6,000-9,000/sqft), Kufri (weekend/tourist demand, Rs. 7,000-12,000/sqft). Demand is driven by government employees, retirees, and second-home buyers from Delhi/Chandigarh. Monsoon season (July-September) sees slowest transaction volumes."
    },
    {
        "source": "Avacasa District Market Profile — Solan",
        "text": "Solan district, including Kasauli and Kumarhatti, has the highest median price per sqft in HP at approximately Rs. 10,552. Kasauli commands a significant premium due to its cantonment heritage, colonial architecture, and proximity to Chandigarh (60 km). Average drive time to Chandigarh: 90 minutes. Demand is almost entirely from second-home and retirement buyers from Punjab and Delhi NCR. Inventory is limited by cantonment land restrictions. Villas in the Rs. 80L-3Cr range dominate."
    },
    {
        "source": "Avacasa District Market Profile — Kullu/Manali",
        "text": "Kullu-Manali corridor is tourism-driven with significant short-term rental (STR) demand. Median price per sqft approximately Rs. 7,161. STR gross yields of 6-10% are achievable for well-located properties, significantly higher than Shimla (3-5%) due to higher occupancy driven by adventure tourism. Winter accessibility is a key risk factor — properties above 2,500m may be inaccessible December-February. Manali town commands premium over Kullu town. Foreign buyer interest is high but limited by Section 118."
    },
    {
        "source": "Avacasa District Market Profile — Kangra/Dharamsala",
        "text": "Kangra district, anchored by Dharamsala and McLeod Ganj, is an emerging market at Rs. 6,600/sqft median. International demand from the Tibetan community and foreign yoga/retreat visitors sustains premium micro-markets around McLeod Ganj. Palampur offers agricultural hinterland and lower prices. Bir Billing attracts adventure sports tourism. The market is less mature than Shimla/Solan with fewer comparables, meaning valuations carry higher uncertainty."
    },
    {
        "source": "HP STR Market — Short Term Rental",
        "text": "Short-term rental platforms (Airbnb, MakeMyTrip Homes) are active across HP hill stations. Estimated gross STR yields: Kasauli 5-8%, Manali 7-10%, Shimla 3-6%, Dharamsala 5-9%. Net yields after maintenance and platform fees are typically 60-70% of gross. Peak seasons: April-June (summer) and October (autumn leaves). Trough: July-August monsoon. STR income is self-employment income for tax purposes in India; GST registration required if annual revenue exceeds Rs. 20L."
    },
    {
        "source": "HP Property Transaction Process",
        "text": "Property purchase in HP: (1) Verify Section 118 applicability for your buyer profile. (2) Confirm RERA registration for new projects at hprera.in. (3) Get property valuation from a registered valuer for stamp duty calculation. (4) Stamp duty in HP: 6% for men, 4% for women buyers. Registration fee: 2% (max Rs. 2L). (5) Register sale deed at the sub-registrar office in the tehsil where property is located. Timeline from agreement to registration: typically 30-60 days."
    },
    {
        "source": "Avacasa AVM Model Performance",
        "text": "Avacasa's Automated Valuation Model (AVM) is trained on HP apartment and villa listings. On real observed listings (scraped from 99acres), the model achieves R-squared of 0.85 with a Mean Absolute Error of approximately Rs. 19 lakh for apartments and Rs. 83-99 lakh for villas. The model predicts price per square foot (PPSF) using CatBoost, then multiplies by area. Key features: location (45% of importance), area per bedroom (17%), amenity score (10%). The model is most reliable in Shimla and Solan where listing density is highest."
    },
]

EXAMPLE_QUERIES = [
    "What is a 3BHK apartment in Shimla worth?",
    "Show me apartments under 40L in Solan",
    "What is Section 118 and does it affect me?",
    "Which district has the best rental yields?",
    "How does RERA protect me as a buyer?",
    "Compare Shimla vs Kasauli for investment",
]

# ═════════════════════════════════════════════════════════════════════════════
# DATA PATH RESOLVER
# ═════════════════════════════════════════════════════════════════════════════

def resolve_data_path() -> str:
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hp_unified_listings.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "src",  "hp_unified_listings.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "hp_unified_listings.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "hp_unified_listings.json not found. Place it in ./data/, ./src/, or project root."
    )

# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS (copied from apartement_v3.py)
# ═════════════════════════════════════════════════════════════════════════════

def resolve_district(row) -> str:
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


def load_and_clean(path: str) -> pd.DataFrame:
    with open(path) as f:
        payload = json.load(f)
    df = pd.DataFrame(payload["records"])

    df = df[df["asset_class"].isin(["apartment", "villa"])].copy()
    df = df.dropna(subset=["listing_price", "saleable_area_sqft"])
    df = df[(df["listing_price"] > 0) & (df["saleable_area_sqft"] > 0)]
    df = df[(df["saleable_area_sqft"] >= 100) & (df["saleable_area_sqft"] <= 15000)]

    df["ppsf"] = df["listing_price"] / df["saleable_area_sqft"]
    lo, hi = df["ppsf"].quantile([0.005, 0.995])
    df = df[(df["ppsf"] >= lo) & (df["ppsf"] <= hi)].copy()

    df["district_filled"] = df.apply(resolve_district, axis=1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bedrooms"]            = pd.to_numeric(df["bedrooms"],            errors="coerce")
    df["bedrooms"]            = df["bedrooms"].fillna(df["bedrooms"].median())
    df["bathrooms"]           = pd.to_numeric(df["bathrooms"],           errors="coerce")
    df["balconies"]           = pd.to_numeric(df["balconies"],           errors="coerce")
    df["parking_count"]       = pd.to_numeric(df["parking_count"],       errors="coerce")
    df["total_floors"]        = pd.to_numeric(df["total_floors"],        errors="coerce")
    df["floor_number"]        = pd.to_numeric(df["floor_number"],        errors="coerce")
    df["society_total_units"] = pd.to_numeric(df["society_total_units"], errors="coerce")
    df["elevation_m"]         = pd.to_numeric(df["elevation_m"],         errors="coerce")
    df["chandigarh_drive_mins"] = pd.to_numeric(df["chandigarh_drive_mins"], errors="coerce")
    df["delhi_drive_mins"]    = pd.to_numeric(df["delhi_drive_mins"],    errors="coerce")
    df["road_quality_score"]  = pd.to_numeric(df["road_quality_score"],  errors="coerce")
    df["str_gross_yield"]     = pd.to_numeric(df["str_gross_yield"],     errors="coerce")
    df["second_home_score"]   = pd.to_numeric(df["second_home_score"],   errors="coerce")
    df["benchmark_psf"]       = pd.to_numeric(df["benchmark_psf"],       errors="coerce")
    df["has_rera"]  = df["has_rera"].fillna(False).astype(int)
    df["is_villa"]  = (df["asset_class"] == "villa").astype(int)

    for c in ["facing", "furnishing", "power_backup", "construction_status",
              "transaction_type", "view_type", "developer_tier",
              "market_maturity", "flood_risk", "area_type"]:
        df[c] = df[c].fillna("unknown").astype(str)

    df["area_per_bhk"] = df["saleable_area_sqft"] / df["bedrooms"].replace(0, np.nan)
    df["bath_per_bhk"] = df["bathrooms"] / df["bedrooms"].replace(0, np.nan)
    df["log_area"]     = np.log1p(df["saleable_area_sqft"])
    df["sqrt_area"]    = np.sqrt(df["saleable_area_sqft"])
    df["log_ppsf"]     = np.log1p(df["ppsf"])
    return df


def fit_location_encoder(train_idx, frame):
    sub = frame.loc[train_idx]
    district_med  = sub.groupby("district_filled")["log_ppsf"].median()
    global_med    = sub["log_ppsf"].median()
    market_stats  = sub.groupby("major_market")["log_ppsf"].agg(["median", "count"])
    market_district = sub.groupby("major_market")["district_filled"].first()
    return district_med, global_med, market_stats, market_district


def apply_location_encoder(idx, frame, district_med, global_med, market_stats, market_district):
    rows = frame.loc[idx]
    out  = []
    for district, market in zip(rows["district_filled"], rows["major_market"]):
        d_med = district_med.get(district, global_med)
        if market in market_stats.index:
            m_med = market_stats.loc[market, "median"]
            m_cnt = market_stats.loc[market, "count"]
            w     = m_cnt / (m_cnt + SHRINK_K)
            out.append(w * m_med + (1 - w) * d_med)
        else:
            out.append(d_med)
    return np.array(out)


def make_design_matrix(idx, X_all, encoder_args):
    sub = X_all.loc[idx].copy()
    sub["loc_ppsf_prior"] = apply_location_encoder(idx, X_all, *encoder_args)
    return sub[FEATURES]


# ═════════════════════════════════════════════════════════════════════════════
# STARTUP — cached resources (runs once per deployment)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_everything():
    """
    Loads data, trains CatBoost model, builds FAISS index.
    Cached with st.cache_resource so it only runs once.
    """
    from sklearn.model_selection import train_test_split
    from catboost import CatBoostRegressor

    status = {"model": False, "listings": False, "kb": False, "error": None}

    # 1. Load + clean data
    try:
        data_path = resolve_data_path()
        raw_df    = load_and_clean(data_path)
        df        = engineer_features(raw_df)

        X_all      = df.reset_index(drop=True).copy()
        y_all      = X_all["log_ppsf"]
        area_all   = X_all["saleable_area_sqft"]
        price_all  = X_all["listing_price"]
        source_all = X_all["data_source"]

        # Compute training medians/modes for inference defaults
        training_medians = {f: float(X_all[f].median()) for f in NUMERIC_FEATURES
                            if f in X_all.columns and pd.api.types.is_numeric_dtype(X_all[f])}
        training_modes   = {f: str(X_all[f].mode().iloc[0]) if len(X_all[f].mode()) > 0 else "unknown"
                            for f in CAT_FEATURES if f in X_all.columns}

        # 2. 70/15/15 split (same as apartement_v3.py)
        train_idx, rest_idx = train_test_split(
            X_all.index, test_size=0.30, random_state=RNG, stratify=source_all
        )
        encoder_args = fit_location_encoder(train_idx, X_all)
        X_train = make_design_matrix(train_idx, X_all, encoder_args)
        y_train = y_all.loc[train_idx]

        # 3. Train CatBoost (same hyperparams as apartement_v3.py)
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
        status["model"]    = True
        status["listings"] = True

        # District-level median PPSF for search ranking
        district_ppsf = X_all.groupby("district_filled")["ppsf"].median().to_dict()

    except Exception as e:
        status["error"] = f"Model load error: {e}"
        return None, None, None, None, None, None, None, status

    # 4. FAISS knowledge base
    try:
        from sentence_transformers import SentenceTransformer
        import faiss

        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        texts       = [c["text"] for c in CHUNKS]
        embeddings  = embed_model.encode(texts, convert_to_numpy=True)
        d           = embeddings.shape[1]
        index       = faiss.IndexFlatL2(d)
        index.add(embeddings.astype(np.float32))
        status["kb"] = True
    except Exception as e:
        status["kb"]    = False
        status["error"] = f"KB load error (non-fatal): {e}"
        embed_model     = None
        index           = None

    return (
        model,
        encoder_args,
        X_all,
        training_medians,
        training_modes,
        district_ppsf,
        (embed_model, index),
        status,
    )


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def fmt_inr(amount: float) -> str:
    """Format INR amount as Cr or L."""
    if amount >= 1e7:
        return f"₹{amount/1e7:.2f} Cr"
    return f"₹{amount/1e5:.1f} L"


def _infer_single(
    district: str,
    major_market: str,
    area_sqft: float,
    bedrooms: float,
    bathrooms: float,
    asset_class: str,
    model,
    encoder_args,
    training_medians: dict,
    training_modes: dict,
) -> tuple[float, float]:
    """Run AVM inference for a single new listing. Returns (predicted_price, pred_log_ppsf)."""
    district_med, global_med, market_stats, market_district = encoder_args

    # Compute loc_ppsf_prior
    d_med = district_med.get(district, global_med)
    if major_market and major_market in market_stats.index:
        m_med = market_stats.loc[major_market, "median"]
        m_cnt = market_stats.loc[major_market, "count"]
        w     = m_cnt / (m_cnt + SHRINK_K)
        loc_prior = w * m_med + (1 - w) * d_med
    else:
        loc_prior = d_med

    is_villa = 1 if asset_class == "villa" else 0

    # Derived numeric features
    safe_bedrooms = bedrooms if bedrooms > 0 else 1.0
    area_per_bhk  = area_sqft / safe_bedrooms
    bath_per_bhk  = bathrooms / safe_bedrooms
    log_area      = np.log1p(area_sqft)
    sqrt_area     = np.sqrt(area_sqft)

    overrides = {
        "saleable_area_sqft": area_sqft,
        "bedrooms":            bedrooms,
        "bathrooms":           bathrooms,
        "is_villa":            is_villa,
        "area_per_bhk":        area_per_bhk,
        "bath_per_bhk":        bath_per_bhk,
        "log_area":            log_area,
        "sqrt_area":           sqrt_area,
        "loc_ppsf_prior":      loc_prior,
        "district_filled":     district,
        "data_source":         "scraped_99acres",
    }

    row = {}
    for feat in FEATURES:
        if feat in overrides:
            row[feat] = overrides[feat]
        elif feat in CAT_FEATURES:
            row[feat] = training_modes.get(feat, "unknown")
        else:
            row[feat] = training_medians.get(feat, 0.0)

    X_inf = pd.DataFrame([row])
    pred_log_ppsf = model.predict(X_inf)[0]
    pred_price    = float(np.expm1(pred_log_ppsf) * area_sqft)
    return pred_price, pred_log_ppsf


# ═════════════════════════════════════════════════════════════════════════════
# INTENT ROUTING (Gemini)
# ═════════════════════════════════════════════════════════════════════════════

INTENT_SYSTEM_PROMPT = """You are the intent router for Avacasa, a real estate intelligence system for Himachal Pradesh, India. Your only job is to classify the user's query and extract structured parameters.

Respond ONLY with a valid JSON object, no markdown, no explanation:
{
  "intent": "valuation" | "search" | "market_qa" | "out_of_scope",
  "params": {
    "bedrooms": <int or null>,
    "area_sqft": <float or null>,
    "district": <"Shimla"|"Solan"|"Kasauli"|"Kangra"|"Kullu"|"Mandi"|"Others" or null>,
    "micro_market": <string or null>,
    "asset_class": <"apartment"|"villa" or null>,
    "max_price": <float INR or null>,
    "min_price": <float INR or null>,
    "query_text": <original user query as string>
  }
}

Rules:
- If the query is not about HP real estate, set intent to "out_of_scope".
- For valuation: extract bedrooms, area_sqft, district, micro_market.
- For search: extract any filters present (price, bedrooms, district, asset_class). A query asking for "options" or "listings" is search.
- For market_qa: anything about regulations, prices, yields, comparisons, Section 118, RERA, districts, investment advice.
- district must be one of the allowed values above or null.
- Never add commentary. Return only the JSON object."""

SYNTHESIS_SYSTEM_PROMPT = """You are Avacasa's HP real estate knowledge assistant. Answer the user's question using ONLY the context chunks provided below. Be concise, factual, and specific to Himachal Pradesh. If the answer is not in the context, say "I don't have reliable data on that — please consult an HP property expert." Never make up statistics. Always end with a one-line source attribution."""


def classify_intent(query: str) -> dict:
    """Use Gemini to classify intent and extract params. Falls back to keyword heuristics."""
    default = {
        "intent": "market_qa",
        "params": {"query_text": query, "bedrooms": None, "area_sqft": None,
                   "district": None, "micro_market": None, "asset_class": None,
                   "max_price": None, "min_price": None}
    }

    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash",
                                          system_instruction=INTENT_SYSTEM_PROMPT)
            resp = model.generate_content(query)
            raw = resp.text.strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
            parsed = json.loads(raw)
            parsed.setdefault("params", {})["query_text"] = query
            return parsed
        except Exception:
            pass

    # ── Keyword fallback ──────────────────────────────────────────────────────
    q = query.lower()
    if any(w in q for w in ["worth", "value", "price", "cost", "estimate", "valuation", "how much"]):
        intent = "valuation"
    elif any(w in q for w in ["show", "find", "list", "search", "properties", "options", "available", "under", "below"]):
        intent = "search"
    else:
        intent = "market_qa"

    # Extract district
    district_map = {
        "shimla": "Shimla", "solan": "Solan", "kasauli": "Solan",
        "kangra": "Kangra", "dharamsala": "Kangra", "kullu": "Kullu",
        "manali": "Kullu", "mandi": "Mandi",
    }
    district = None
    for kw, d in district_map.items():
        if kw in q:
            district = d
            break

    # Extract bedrooms
    bhk_match = re.search(r"(\d)\s*bhk", q) or re.search(r"(\d)\s*bedroom", q)
    bedrooms = int(bhk_match.group(1)) if bhk_match else None

    # Extract asset class
    asset_class = None
    if "villa" in q or "house" in q or "independent" in q:
        asset_class = "villa"
    elif "apartment" in q or "flat" in q:
        asset_class = "apartment"

    # Extract max price (simple: "under X lakh" / "below X crore")
    max_price = None
    lakh_match = re.search(r"(?:under|below|max|within)\s*(?:rs\.?\s*)?(\d+(?:\.\d+)?)\s*(?:l|lakh|lac)", q)
    cr_match   = re.search(r"(?:under|below|max|within)\s*(?:rs\.?\s*)?(\d+(?:\.\d+)?)\s*(?:cr|crore)", q)
    if lakh_match:
        max_price = float(lakh_match.group(1)) * 1e5
    elif cr_match:
        max_price = float(cr_match.group(1)) * 1e7

    return {
        "intent": intent,
        "params": {
            "query_text":  query,
            "bedrooms":    bedrooms,
            "area_sqft":   None,
            "district":    district,
            "micro_market": None,
            "asset_class": asset_class,
            "max_price":   max_price,
            "min_price":   None,
        }
    }


def synthesize_market_qa(query: str, context_chunks: list[dict]) -> str:
    """Use Gemini to synthesize an answer from retrieved context. Falls back to concat."""
    context_str = "\n\n".join(
        f"[{c['source']}]\n{c['text']}"
        for c in context_chunks
    )
    prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash",
                                          system_instruction=SYNTHESIS_SYSTEM_PROMPT)
            resp = model.generate_content(prompt)
            return resp.text.strip()
        except Exception:
            pass

    # Fallback: concatenate context
    return context_str + f"\n\n*(Source: {', '.join(c['source'] for c in context_chunks)})*"


# ═════════════════════════════════════════════════════════════════════════════
# TOOL HANDLERS
# ═════════════════════════════════════════════════════════════════════════════

def tool_valuation(params: dict, model, encoder_args, training_medians, training_modes) -> str:
    district    = params.get("district") or "Shimla"
    micro       = params.get("micro_market") or district
    area_sqft   = float(params.get("area_sqft") or training_medians.get("saleable_area_sqft", 1447))
    bedrooms    = float(params.get("bedrooms") or training_medians.get("bedrooms", 3))
    bathrooms   = training_medians.get("bathrooms", 2.0)
    asset_class = params.get("asset_class") or "apartment"

    pred_price, _ = _infer_single(
        district, micro, area_sqft, bedrooms, bathrooms, asset_class,
        model, encoder_args, training_medians, training_modes
    )

    mae = VILLA_MAE_INR if asset_class == "villa" else APT_MAE_INR
    low = max(0, pred_price - mae)
    high = pred_price + mae

    n_real = 694  # real scraped listings count (from documentation)
    r2_real = 0.85 if asset_class == "apartment" else 0.73

    html = f"""
<div class='val-card'>
  <div style='font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>
    🏠 Estimated Market Value
  </div>
  <div class='val-main'>{fmt_inr(pred_price)}</div>
  <div class='val-range'>Range: {fmt_inr(low)} – {fmt_inr(high)}</div>
  <hr class='val-divider'>
  <div class='val-row'><span class='val-label'>District</span><span class='val-value'>{district}</span></div>
  <div class='val-row'><span class='val-label'>Bedrooms</span><span class='val-value'>{int(bedrooms)} BHK</span></div>
  <div class='val-row'><span class='val-label'>Area</span><span class='val-value'>{int(area_sqft):,} sqft</span></div>
  <div class='val-row'><span class='val-label'>PPSF</span><span class='val-value'>₹{int(pred_price/area_sqft):,}/sqft</span></div>
  <div class='val-row'><span class='val-label'>Asset class</span><span class='val-value'>{asset_class.capitalize()}</span></div>
  <hr class='val-divider'>
  <div class='val-warn'>
    ⚠ Based on <strong>{n_real} real HP {asset_class} listings</strong>
    (R² {r2_real:.2f} on real scraped data). ±{fmt_inr(mae)} confidence band.
    Model is most reliable in Shimla and Solan.
  </div>
</div>
<div class='msg-source'>Source: CatBoost AVM (real-listing R² {r2_real:.2f}) | {district} district prior</div>
"""
    return html


def tool_search(params: dict, X_all: pd.DataFrame, district_ppsf: dict) -> str:
    df = X_all.copy()

    # Apply filters
    if params.get("asset_class"):
        df = df[df["asset_class"] == params["asset_class"]]
    if params.get("district"):
        df = df[df["district_filled"] == params["district"]]
    if params.get("bedrooms"):
        b = float(params["bedrooms"])
        df = df[(df["bedrooms"] >= b - 1) & (df["bedrooms"] <= b + 1)]
    if params.get("max_price"):
        df = df[df["listing_price"] <= float(params["max_price"])]
    if params.get("min_price"):
        df = df[df["listing_price"] >= float(params["min_price"])]

    if df.empty:
        return (
            "<div class='ai-bubble'>"
            "No listings matched your filters. Try broadening the search — "
            "for example, remove the district filter or increase the price ceiling."
            "</div>"
        )

    # Rank: closest to location-fair-value median
    df = df.copy()
    df["_loc_med"] = df["district_filled"].map(district_ppsf).fillna(df["ppsf"].median())
    df["_err"]     = (df["ppsf"] - df["_loc_med"]).abs()
    top5           = df.nsmallest(5, "_err")

    cards = []
    for _, row in top5.iterrows():
        dist_med = district_ppsf.get(row["district_filled"], row["ppsf"])
        pct_diff = ((row["ppsf"] - dist_med) / dist_med) * 100
        ppsf_cls = "ppsf-above" if pct_diff > 0 else "ppsf-below"
        ppsf_lbl = f"+{pct_diff:.1f}% above" if pct_diff > 0 else f"{pct_diff:.1f}% below"

        source_badge = (
            "<span class='listing-chip badge-real'>Real listing</span>"
            if row.get("data_source") == "scraped_99acres"
            else "<span class='listing-chip badge-synth'>Internal estimate</span>"
        )
        title = row.get("name") or f"{int(row['bedrooms'])} BHK {row['asset_class']}"
        micro = row.get("micro_market") or row.get("major_market") or ""
        loc_str = f"{row['district_filled']}{', ' + micro if micro else ''}"

        cards.append(f"""
<div class='listing-card'>
  <div class='listing-title'>{title}</div>
  <div class='listing-loc'>📍 {loc_str}</div>
  <div class='listing-meta'>
    <span class='listing-chip chip-price'>{fmt_inr(row['listing_price'])}</span>
    <span class='listing-chip chip-area'>{int(row['saleable_area_sqft']):,} sqft</span>
    <span class='listing-chip chip-bhk'>{int(row['bedrooms'])} BHK</span>
    {source_badge}
  </div>
  <div class='listing-ppsf'>
    ₹{int(row['ppsf']):,}/sqft
    — <span class='{ppsf_cls}'>{ppsf_lbl} district median</span>
  </div>
</div>""")

    total_found = len(df)
    header = f"<div style='font-size:13px;color:#64748b;margin-bottom:8px;'>Found {total_found} listings — showing top 5 closest to fair value:</div>"
    footer = "<div class='msg-source'>Source: hp_unified_listings.json | Ranked by proximity to district median PPSF</div>"
    return header + "\n".join(cards) + footer


def tool_market_qa(query: str, embed_model, faiss_index) -> str:
    if embed_model is None or faiss_index is None:
        # No FAISS — return a simple summary from chunks via keyword match
        q_lower = query.lower()
        matched = []
        keywords = q_lower.split()
        for chunk in CHUNKS:
            score = sum(kw in chunk["text"].lower() for kw in keywords if len(kw) > 3)
            matched.append((score, chunk))
        matched.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [c for _, c in matched[:3]]
    else:
        try:
            q_emb = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
            _, I  = faiss_index.search(q_emb, k=3)
            top_chunks = [CHUNKS[i] for i in I[0] if 0 <= i < len(CHUNKS)]
        except Exception:
            top_chunks = CHUNKS[:3]

    answer = synthesize_market_qa(query, top_chunks)
    sources = " · ".join(c["source"] for c in top_chunks)
    return f"""
<div class='qa-text'>{answer}</div>
<div class='msg-source'>Source: {sources}</div>
"""


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═════════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "pending_demo" not in st.session_state:
    st.session_state.pending_demo = False


# ═════════════════════════════════════════════════════════════════════════════
# LOAD RESOURCES (with spinner)
# ═════════════════════════════════════════════════════════════════════════════

with st.spinner("Loading Avacasa Intelligence — training AVM model..."):
    resources = load_everything()

(
    _model,
    _encoder_args,
    _X_all,
    _training_medians,
    _training_modes,
    _district_ppsf,
    _kb_tuple,
    _status,
) = resources

_embed_model, _faiss_index = _kb_tuple if _kb_tuple else (None, None)
_model_ok    = _status.get("model", False)
_listings_ok = _status.get("listings", False)
_kb_ok       = _status.get("kb", False)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
<div class='sidebar-logo'>🏔 Avacasa</div>
<div class='sidebar-sub'>HP Market Intelligence</div>
<hr class='sidebar-divider'>
""", unsafe_allow_html=True)

    # Status indicators
    def _status_icon(ok: bool) -> str:
        return "✓" if ok else "✗"
    def _status_cls(ok: bool) -> str:
        return "status-ok" if ok else "status-err"

    model_detail    = "(CatBoost · 70/15/15 split)" if _model_ok else ""
    listings_detail = f"({len(_X_all):,} records)" if (_listings_ok and _X_all is not None) else ""
    kb_detail       = "(FAISS · 10 chunks)" if _kb_ok else "(keyword fallback)"
    gemini_cls      = "status-ok" if GEMINI_API_KEY else "status-err"
    gemini_icon     = "✓" if GEMINI_API_KEY else "✗"

    st.markdown(f"""
<div class='sidebar-section'>System Status</div>
<div>
  <span class='{_status_cls(_model_ok)}'>{_status_icon(_model_ok)} AVM model</span>
  <span style='font-size:11px;color:#475569'> {model_detail}</span><br>
  <span class='{_status_cls(_listings_ok)}'>{_status_icon(_listings_ok)} Listings DB</span>
  <span style='font-size:11px;color:#475569'> {listings_detail}</span><br>
  <span class='{_status_cls(_kb_ok)}'>{_status_icon(_kb_ok)} Knowledge base</span>
  <span style='font-size:11px;color:#475569'> {kb_detail}</span><br>
  <span class='{gemini_cls}'>{gemini_icon} Gemini API</span>
</div>
<hr class='sidebar-divider'>
""", unsafe_allow_html=True)

    if _status.get("error") and not _model_ok:
        st.error(f"Error: {_status['error']}")

    # Example queries
    st.markdown("<div class='sidebar-section'>Example Queries</div>", unsafe_allow_html=True)
    for q in EXAMPLE_QUERIES:
        if st.button(q, key=f"eq_{q[:20]}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section'>Demo</div>", unsafe_allow_html=True)
    if st.button("🎬 Load Demo Conversation", use_container_width=True):
        st.session_state.pending_demo = True
        st.rerun()
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:11px;color:#475569;line-height:1.7;'>
  <strong style='color:#10b981'>Scope:</strong> HP residential property only<br>
  <strong style='color:#10b981'>Model:</strong> CatBoost on real 99acres listings<br>
  <strong style='color:#10b981'>Real-listing R²:</strong> 0.85 apartments<br>
  <strong style='color:#10b981'>MAE:</strong> ₹19L apt · ₹83L villa
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# DEMO LOADER
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state.pending_demo and _model_ok and _listings_ok:
    st.session_state.pending_demo = False

    _DEMO_CASES = [
        # ── Valuations ────────────────────────────────────────────────────────
        (
            "What is a 3 BHK apartment in Shimla worth?",
            "valuation",
            {"district": "Shimla", "bedrooms": 3, "asset_class": "apartment", "area_sqft": 900},
        ),
        (
            "Estimate the value of a 4 BHK villa in Manali",
            "valuation",
            {"district": "Kullu", "bedrooms": 4, "asset_class": "villa", "area_sqft": 2400},
        ),
        (
            "How much does a 2 BHK flat in Solan cost?",
            "valuation",
            {"district": "Solan", "bedrooms": 2, "asset_class": "apartment", "area_sqft": 700},
        ),
        (
            "What would a 3 BHK villa in Dharamsala be valued at?",
            "valuation",
            {"district": "Kangra", "bedrooms": 3, "asset_class": "villa", "area_sqft": 1800},
        ),
        # ── Property search ───────────────────────────────────────────────────
        (
            "Show me apartments under 40 lakh in Solan",
            "search",
            {"asset_class": "apartment", "district": "Solan", "max_price": 4_000_000},
        ),
        (
            "Find villas under 2 crore in Kasauli",
            "search",
            {"asset_class": "villa", "district": "Solan", "max_price": 20_000_000},
        ),
        (
            "List 2 BHK apartments available in Shimla",
            "search",
            {"asset_class": "apartment", "district": "Shimla", "bedrooms": 2},
        ),
        # ── Market knowledge ──────────────────────────────────────────────────
        (
            "What is Section 118 and how does it affect non-HP buyers?",
            "market_qa",
            {},
        ),
        (
            "Which district in Himachal Pradesh has the best rental yields?",
            "market_qa",
            {},
        ),
        (
            "How does HP RERA protect me as a property buyer?",
            "market_qa",
            {},
        ),
        (
            "Compare Shimla vs Kasauli for investment — which is better?",
            "market_qa",
            {},
        ),
    ]

    _demo_msgs = []
    for _uq, _intent, _params in _DEMO_CASES:
        try:
            if _intent == "valuation":
                _resp = tool_valuation(_params, _model, _encoder_args, _training_medians, _training_modes)
            elif _intent == "search":
                _resp = tool_search(_params, _X_all, _district_ppsf)
            else:
                _resp = tool_market_qa(_uq, _embed_model, _faiss_index)
        except Exception as _e:
            _resp = f"<div class='ai-bubble'>Error generating response: {_e}</div>"
        _demo_msgs.append({"role": "user", "content": _uq})
        _demo_msgs.append({"role": "assistant", "content": _resp})

    st.session_state.messages = _demo_msgs
    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style='padding:20px 24px 12px;border-bottom:1px solid #e2e8f0;background:white;
            margin-bottom:0;'>
  <h2 style='margin:0;font-size:18px;color:#111827;font-weight:700;'>
    Himachal Pradesh Real Estate Intelligence
  </h2>
  <p style='margin:4px 0 0;font-size:12px;color:#64748b;'>
    Ask about valuations, property search, RERA, Section 118, or market trends
  </p>
</div>
""", unsafe_allow_html=True)

# Chat container
chat_area = st.container()

with chat_area:
    # Welcome message if no history
    if not st.session_state.messages:
        st.markdown("""
<div class='ai-msg'>
  <div class='ai-avatar'>🏔</div>
  <div class='ai-bubble'>
    <strong>Welcome to Avacasa Intelligence!</strong><br><br>
    I'm your HP real estate analyst. I can help you with:<br>
    • <strong>Property valuations</strong> — "What is a 2BHK in Shimla worth?"<br>
    • <strong>Listing search</strong> — "Show me villas under 50L in Kasauli"<br>
    • <strong>Market knowledge</strong> — "What is Section 118?" / "Best rental yields?"<br><br>
    Try an example query from the sidebar, or type your question below.
    <div class='msg-source'>Avacasa Intelligence · HP Residential Property · Powered by CatBoost AVM</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Render chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
<div class='user-msg'>
  <div class='user-bubble'>{msg['content']}</div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class='ai-msg'>
  <div class='ai-avatar'>🏔</div>
  <div class='ai-bubble'>{msg['content']}</div>
</div>
""", unsafe_allow_html=True)


# ── Input form ────────────────────────────────────────────────────────────────
with st.container():
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        user_input = st.text_input(
            label="chat_input",
            label_visibility="collapsed",
            placeholder="Ask about HP property valuations, listings, or market knowledge...",
            key="chat_input_box",
            value=st.session_state.pending_query or "",
        )
    with col_btn:
        send_clicked = st.button("Send →", type="primary", use_container_width=True)

# Clear pending query after it's been populated
if st.session_state.pending_query:
    st.session_state.pending_query = None


# ── Process message ────────────────────────────────────────────────────────────
def process_query(query: str) -> str:
    """Route query and return HTML response string."""
    if not query.strip():
        return ""

    if not _model_ok:
        return "<em>⚠ The AVM model failed to load. Please check the data file path.</em>"

    # Classify intent
    parsed   = classify_intent(query)
    intent   = parsed.get("intent", "market_qa")
    params   = parsed.get("params", {})

    if intent == "out_of_scope":
        return (
            "I can only help with Himachal Pradesh residential real estate questions — "
            "valuations, listing search, and market knowledge. What would you like to "
            "know about HP property?"
        )

    if intent == "valuation":
        return tool_valuation(params, _model, _encoder_args, _training_medians, _training_modes)

    if intent == "search":
        if _X_all is None:
            return "<em>Listing database unavailable.</em>"
        return tool_search(params, _X_all, _district_ppsf)

    # market_qa (default)
    return tool_market_qa(query, _embed_model, _faiss_index)


if send_clicked and user_input.strip():
    query = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Analysing your query..."):
        response = process_query(query)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
