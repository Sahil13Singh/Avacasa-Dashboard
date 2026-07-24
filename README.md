# Avacasa Intelligence Dashboard

Conversational real estate intelligence for Himachal Pradesh (HP) residential property. Ask about property valuations, search listings, or get answers on HP-specific regulations (RERA, Section 118) and market trends — all in one Streamlit chat interface.

**Live app:** [avacasa-dashboard-fcohq2cvbbsuib5v7zvxau.streamlit.app](https://avacasa-dashboard-fcohq2cvbbsuib5v7zvxau.streamlit.app)

## What it does

- **Valuation (AVM)** — estimates market value for an apartment/villa given district, bedrooms, and area, using a CatBoost model trained on HP listings.
- **Property search** — filters the listings database by district, price, bedrooms, and asset class.
- **Market Q&A** — retrieval-augmented answers (FAISS + sentence-transformers) grounded in a small knowledge base covering HP RERA, Section 118 land law, district market profiles, and short-term rental yields, synthesized with Gemini.

Queries are routed automatically (via Gemini, with a keyword-based fallback) to whichever of the three tools above fits the question.

## Project layout

```
streamlit_app.py               # the deployed app — UI, routing, valuation, search, and Q&A
data/hp_unified_listings.json  # listings dataset the app loads at startup
apartement_v3.py               # standalone AVM training/evaluation script (apartments)
villa.py                       # standalone AVM training/evaluation script (villas)
requirements.txt                # dependencies Streamlit Cloud installs from
runtime.txt                     # pins the Python version for Streamlit Cloud
```

`apartement_v3.py` and `villa.py` are exploratory/training scripts used to develop the model — the deployed app trains its own model in-process from `data/hp_unified_listings.json` at startup (see `load_everything()` in `streamlit_app.py`).

## Running locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Create a `.env` file in the project root with your Gemini key:

```
GEMINI_API_KEY=your-key-here
```

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (already set up).
2. On [share.streamlit.io](https://share.streamlit.io), point the app at `streamlit_app.py`.
3. In the app's **Settings → Secrets**, add:

   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```

4. Streamlit Cloud installs from `requirements.txt` and uses the Python version in `runtime.txt` automatically.

## Notes on the data

`hp_unified_listings.json` mixes real scraped listings (99acres) with synthetic Avacasa-generated records for coverage — synthetic rows are flagged `is_synthetic: true` and are not observed transactions. See the caveat at the top of `villa.py` for details on the split and known limitations.
