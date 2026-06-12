# HouseAccount AI Pricing Model

An AI-powered pricing model for HouseAccount's home services marketplace. Takes a booking request, extracts scope signals from the job description using Claude Haiku, and returns a price estimate with a calibrated confidence score.

**Results vs. baseline (`original_estimate` column):**

| Benchmark | Baseline | This model | Result |
|-----------|----------|------------|--------|
| Blended MAPE (411 priced rows) | 11.6% | 11.4% (routed) | ✓ beats |
| Handyman MAPE (real-only target) | 48.4% | 36.2% (blended) | ✓ beats ~40% target |
| Response time | — | 5–11ms warm | ✓ under 2s |

---

## How it works

The baseline model ignores `job_description` entirely. Its worst failures are all Handyman jobs where scope complexity lives in the text: "install supplied shutters" was priced at $750 when the actual was $250 — a pure labor job, no materials included.

This model fixes that with two layers:

1. **Claude Haiku** extracts four scope features from every job description: `labor_only`, `task_count`, `complexity_tier`, `has_area_measure`. These are the signals the baseline can't see.
2. **XGBoost quantile regression** trains on those features + category + state median income. At serving time it routes by category: well-priced categories (HVAC, Cleaning, Moving, Landscaping, Pest Control, Roofing) pass through `original_estimate` directly; hard categories (Handyman, Plumbing, Flooring, Appliance Repair) use the model.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Scope extraction | Claude Haiku (Anthropic) via tool-use |
| Pricing model | XGBoost 2.1 quantile regression (Python) |
| Model service | FastAPI + uvicorn |
| API facade | Ruby on Rails 8.1 (API mode) |
| Tests | Rails Minitest + WebMock |

---

## Prerequisites

- Python 3.9+
- Ruby 3.3+ with Bundler
- `brew install libomp` (macOS — required for XGBoost)
- An Anthropic API key

---

## Setup (under 15 minutes)

### 1. Clone and install

```bash
git clone https://github.com/heilashahidi/ai-pricing-model.git
cd ai-pricing-model
pip install anthropic fastapi uvicorn xgboost scikit-learn joblib pgeocode pandas
cd pricing_api && bundle install && cd ..
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```
ANTHROPIC_API_KEY=sk-ant-...
GAUNTLET_PRICING_SECRET=choose-any-secret-string
```

The HouseAccount staging credentials (`HA_APP_NAME`, `HA_SIGNING_SECRET`) are already set in `.env.example`.

### 3. Build the model

Run these three scripts in order. Each is resumable if interrupted.

```bash
# Extract scope features from all 1,432 job descriptions (~5–8 min, ~$0.60)
python3 extract_features.py

# Enrich with state-level income data (offline, ~10 sec)
python3 enrich_zip.py

# Train XGBoost models and evaluate (~3 min)
python3 train.py
```

`train.py` prints a full evaluation report and saves artifacts to `models/`.

### 4. Start the services

**Terminal 1 — FastAPI model service:**

```bash
python3 -m uvicorn pricing_service:app --port 8000
```

**Terminal 2 — Rails API:**

```bash
cd pricing_api
GAUNTLET_PRICING_SECRET=<your-secret> bundle exec rails server -p 3000
```

### 5. Verify

```bash
python3 integration_test.py
```

Expected output: `13/13 passed | 0 failed`

---

## Making a request

```bash
curl -X POST http://localhost:3000/.netlify/functions/pricing-estimate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <GAUNTLET_PRICING_SECRET>" \
  -d '{
    "job_id": "abc123",
    "service_category": "Handyman",
    "zip_code": "33484",
    "job_description": "Install 3 supplied exterior shutters (we supply all hardware)",
    "deadline": "Within 1-2 weeks",
    "original_estimate": 750
  }'
```

Response:

```json
{
  "ok": true,
  "job_id": "abc123",
  "estimate_lo": 343.02,
  "estimate_hi": 525.57,
  "estimate_midpoint": 427.25,
  "confidence": 0.70,
  "model_version": "heila-v1.0.0"
}
```

---

## Project structure

```
.
├── extract_features.py      # Batch Haiku extraction → features.csv
├── enrich_zip.py            # ZIP → state income enrichment → features_enriched.csv
├── train.py                 # XGBoost training + evaluation
├── pricing_service.py       # FastAPI serving layer (port 8000)
├── validate_extraction.py   # Spot-check Haiku extraction quality
├── integration_test.py      # End-to-end tests incl. HouseAccount staging
├── models/
│   ├── meta.json            # Feature config, category lists, OOD thresholds
│   └── *.joblib             # Trained model artifacts (regenerate with train.py)
├── pricing_api/             # Rails 8 API facade (port 3000)
│   ├── app/controllers/pricing_controller.rb
│   ├── config/routes.rb
│   └── test/controllers/pricing_controller_test.rb
├── houseaccount_pricing_sample.csv
├── .env.example
└── notes.md
```

---

## Running tests

**Rails controller tests (12 tests):**

```bash
cd pricing_api
GAUNTLET_PRICING_SECRET=test-secret bundle exec rails test
```

**Full integration test (13 tests, requires both services running):**

```bash
python3 integration_test.py
```

---

## Modeling approach

### Why the baseline fails

The `original_estimate` column achieves 11.6% blended MAPE but 48.4% on Handyman. Every worst prediction is a scope mismatch: labor-only jobs (customer supplies all parts) were priced as if materials were included. The description text contains the signal; the model was blind to it.

### Feature extraction

Claude Haiku extracts four features per job description via tool-use (structured JSON output, no parsing failures):

| Feature | Type | Signal |
|---------|------|--------|
| `labor_only` | bool | Customer supplies parts → 40–60% cheaper |
| `task_count` | int | Number of distinct tasks in the job |
| `complexity_tier` | low/med/high | Single task vs. multi-trade scope |
| `has_area_measure` | bool | Square footage or room count mentioned |

Validation: 14/14 correct on the priced Handyman rows before training.

### Model

Three independent XGBoost regressors with `objective='reg:quantileerror'`:
- q=0.1 → `estimate_lo`
- q=0.5 → `estimate_midpoint`
- q=0.9 → `estimate_hi`

Post-prediction interval crossing is corrected: `lo = min(lo, mid)`, `hi = max(hi, mid)`.

### Routing strategy

For well-priced categories (Cleaning, HVAC, Landscaping, Moving, Pest Control, Roofing), the baseline achieves 8–10% MAPE with 65–95 priced rows each. The model is passed through `original_estimate` directly for these. For hard categories (Handyman, Plumbing, Flooring, Appliance Repair), the XGBoost model is used with a 70/30 blend.

### Confidence calibration

```
base = 1 / (1 + (estimate_hi - estimate_lo) / estimate_midpoint)
confidence = min(base, ood_cap)
```

OOD caps (confidence ≤ 0.40–0.45):
- `estimate_midpoint > $5,000`
- Interval width > 3× median training interval ($212)
- `service_category` outside the 8 production verticals

---

## Regenerating from scratch

If `features.csv` or `models/` are missing:

```bash
python3 extract_features.py   # ~$0.60 in Anthropic API cost
python3 enrich_zip.py
python3 train.py
```

The extraction script is resumable — it skips rows already in `features.csv`.
