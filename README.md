# AutoBusinessGPT

**Upload any business dataset. Let AI do the rest.**

AutoBusinessGPT ingests an arbitrary business CSV or Excel file, figures out what
the columns *mean*, cleans the data, builds a queryable database, runs the machine
-learning models the data can actually support, and explains the results — with a
dashboard, a natural-language SQL copilot, and a downloadable PDF report.

It is built around one honest principle: **it does what your data supports, and
tells you what it can't.** A dataset with no customer column gets no churn model —
and the app says so, instead of inventing one.

---

## What makes it "auto"

The hard problem in "analyze any dataset" is that every dataset has different
column names. AutoBusinessGPT solves this with a **schema-detection engine** that
maps each column to a fixed vocabulary of ~24 *semantic roles* (customer_id,
revenue, order_date, product, …). Everything downstream speaks roles, not column
names, so the same pipeline works on a retail export, a café's till data, or a
SaaS billing table.

Detection is **Gemini-first with a heuristic fallback**:

- With a Gemini API key, an LLM reads column names *and* sample values and maps
  them — handling vocabulary like `TransactionAmount → revenue` or `MRR → revenue`
  that rules would miss.
- Without a key, a built-in heuristic detector (name patterns + value signals:
  cardinality, datetime-parseability, numeric magnitude, email/phone shape) does
  the job offline. It's weaker on unusual vocabulary, which is exactly why you
  then get to…
- **Confirm before analyzing.** Every run shows you the detected roles and
  business type on a review screen where you correct anything wrong. Low-confidence
  guesses are flagged. Detection imperfection becomes a 10-second review, not a
  silent error.

## Pipeline

```
Upload → Detect schema → [Confirm] → Validate quality → Clean →
Build SQLite → Engineer features → Train supported models → Generate insights
```

Each model is **gated on roles**, not guesswork:

| Model | Runs when… |
|-------|------------|
| Sales forecast | there's a date + revenue column and ≥6 months of history |
| Customer churn | there's customer + date + revenue and ≥50 customers with repeat purchases |
| Segmentation | there's customer + date + revenue and ≥50 customers |

What can't run is reported with the reason (e.g. *"churn skipped: missing customer_id"*).

## Features

- **Dynamic dashboard** — charts adapt to what was produced (revenue trend,
  forecast with confidence band, segments, churn risk, feature importance,
  correlation matrix). Panels that can't be filled explain why.
- **Business Copilot** — ask questions in plain English. Gemini writes SQL,
  which runs through a **read-only database engine hardened with a SQLite
  authorizer** — a prompt-injected `DROP TABLE` fails at the driver, not by
  keyword-grepping.
- **Data quality + cleaning** — role-aware checks (negative revenue, future
  dates, impossible discounts) with a full log of every change made.
- **PDF report** — cover, executive summary, KPIs, insights, model results,
  recommendations.
- **Document Chat (optional)** — RAG over uploaded PDFs via FAISS +
  sentence-transformers. Degrades gracefully if those heavy deps aren't installed.

## Install

```bash
# 1. Clone / unzip, then from the project root:
python -m venv .venv && source .venv/bin/activate    # optional
pip install -r requirements.txt

# 2. (Recommended) add a Gemini API key
cp .env.example .env
# edit .env and set GEMINI_API_KEY=...   (get one at https://aistudio.google.com/apikey)

# 3. Run
streamlit run app.py
```

Open http://localhost:8501 and upload a file — or start with one of the bundled
examples in `sample_data/`.

### Without an API key

The app runs fully offline: heuristic schema detection, template-based narratives,
and all charts/models still work. The Copilot and Document Chat need a key (they
require live generation) and will say so.

### Optional Document Chat

RAG pulls in `torch` via `sentence-transformers` (~2 GB). It's commented out in
`requirements.txt`. Enable it with:

```bash
pip install sentence-transformers faiss-cpu pypdf
```

## Docker

```bash
docker build -t autobusinessgpt .
docker run -p 8501:8501 --env-file .env autobusinessgpt
```

## Sample data

| File | Shape | Demonstrates |
|------|-------|--------------|
| `retail_sales.csv` | 1.5k × 21 | full pipeline — all three models run |
| `cafe_transactions.csv` | 2k × 8 | different vocabulary, detection generalising |
| `saas_accounts.csv` | 800 × 6 | partial features — some models correctly skipped |

## Project layout

```
core/          config, semantic-role vocabulary, theme
detection/     schema.py (result types), heuristic + gemini detectors, unified entry
pipeline/      loader, quality, cleaner, database, features, insights, runner
ml/            orchestrator (role-gating) + forecasting, churn, segmentation
dashboard/     charts, components, results view
sql_agent/     NL→SQL copilot (read-only + authorizer boundary)
rag/           document chat (FAISS, graceful degradation)
report/        ReportLab PDF builder
utils/         logger, helpers, Gemini client (google-genai SDK)
app.py         Streamlit entry point
```

## Configuration

All via environment variables (see `.env.example`). Notable ones:

- `GEMINI_API_KEY` — enables all AI features
- `DETECTION_STRATEGY` — `gemini_first` (default) · `heuristic_only` · `gemini_only`
- `DETECTION_CONFIRM_THRESHOLD` — confidence below which a column is flagged (0.75)
- `FORECAST_HORIZON`, `MIN_ROWS_FOR_ML` — model tunables

## Notes & honest limitations

- **The churn label is behavioural**, derived from recency vs each customer's own
  purchase gap (no dataset ships a churn flag). Reported AUC can look very high
  because the label correlates with the recency feature — treat it as "who matches
  the disengagement pattern," not a validated ground-truth predictor.
- **Business type is contextual only.** It's shown and confirmable, but it never
  gates a model — roles do. A mislabeled business type can't switch off analysis
  the data supports.
- **Heuristic detection is a fallback**, not the star. On unusual column names it
  will miss mappings; the confirm screen is how you fix them. Use a Gemini key for
  best detection.
- The Streamlit UI is the least unit-testable layer; core logic (detection,
  cleaning, DB security, models, charts, report) is exercised directly.

## Tech stack

Streamlit · pandas · scikit-learn · XGBoost · SQLAlchemy/SQLite · Plotly ·
Google Gemini (`google-genai`) · ReportLab · FAISS + sentence-transformers (optional)
