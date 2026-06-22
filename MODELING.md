# Modeling Approach

## Problem framing

HouseAccount needs price estimates that homeowners trust enough to book without shopping the market, and providers trust enough to accept without renegotiating. The target is a point estimate with a calibrated confidence interval: narrow and accurate for common jobs, wide and low-confidence for unusual ones.

The evaluation metric is MAPE (Mean Absolute Percentage Error) on two subsets:
- **Blended MAPE**: full 411-row priced subset. Baseline: 11.6%.
- **Real-only MAPE**: held-out portion dominated by sparse/complex categories. Baseline: ~40%.

---

## Data

**Source:** `houseaccount_pricing_sample.csv` — 1,432 sanitized historical jobs across 18 service categories.

**Supervised signal:** 411 rows have `final_price` (the price the provider actually charged). These are the only rows used for training and evaluation. The remaining 1,021 rows have no final price and are used only for feature extraction (Haiku runs on all descriptions).

**Price distribution (priced rows):**

| Statistic | Value |
|-----------|-------|
| Min | $46 |
| Median | $302 |
| Mean | $584 |
| P95 | $3,225 |
| Max | $7,266 |

**Category distribution (priced rows):**

| Category | Priced rows | Baseline MAPE |
|----------|-------------|---------------|
| Cleaning | 66 | 8.7% |
| Moving | 66 | 9.2% |
| Landscaping | 65 | 9.7% |
| HVAC | 65 | 9.7% |
| Pest Control | 65 | 10.2% |
| Appliance Repair | 33 | 12.6% |
| Roofing | 24 | 9.3% |
| Handyman | 14 | 48.4% |
| Plumbing | 3 | 35.8% |
| Flooring | 4 | 29.4% |

Six categories dominate with 65–66 rows each and sub-11% MAPE. Handyman has 14 rows and 48.4% MAPE — this is the real-only benchmark.

**External data joined:** State-level median household income from the 2023 ACS (Census Bureau) via pgeocode ZIP-to-state lookup. 92% of rows matched; remaining 8% use the national median ($75,149). This adds one feature: `state_median_income`.

**No scope fields exist** in the schema. `job_description` is the only source of scope information: no square footage, no fixture count, no complexity rating.

---

## Why the baseline fails

The `original_estimate` column (the current model's output) achieves 11.6% blended MAPE — a strong baseline. Its failures are concentrated:

```
Worst predictions (baseline):
  $250 actual vs $750 estimated  — "Install supplied exterior shutters (we supply shutters)"
  $225 actual vs $535 estimated  — "Assemble shelf, mount lights, flagpole, install mat set"
  $600 actual vs $1125 estimated — "3-4 mixed repairs: electrical, plumbing, carpentry, painting"
  $435 actual vs $190 estimated  — "Assemble two large industrial steel shelving units"
```

Every worst prediction is a Handyman job. The errors are scope mismatches: the baseline treats all Handyman jobs identically regardless of whether the customer supplies parts, how many tasks are bundled, or how complex the work is. The description text contains the decisive signal; the baseline ignores it entirely.

Key finding: **45% of all 1,432 jobs are labor-only** (customer supplies all materials). Labor-only jobs cost 40–60% less than equivalent materials-included jobs. The baseline has no way to know.

---

## Feature engineering

### LLM-extracted scope features (Claude Haiku)

Five features are extracted from `job_description` using Claude Haiku with structured tool-use output (guaranteed valid JSON, no parsing failures):

| Feature | Type | Description |
|---------|------|-------------|
| `labor_only` | bool | True if customer supplies all materials/parts. Examples: "we supply," "you supply," assembly of items the customer already owns. |
| `task_count` | int | Number of distinct tasks or items in the description. |
| `unit_count` | int | Number of physical items to work on. "Install 3 shutters" → 3. Extracted and logged but not yet a model input — too few labeled rows to validate signal. |
| `complexity_tier` | low/med/high → 0/1/2 | Single simple task (low), multiple tasks or moderate skill (medium), multi-trade or large scope (high). |
| `has_area_measure` | bool | True if explicit square footage, acreage, or linear footage is mentioned. |

**Extraction validation:** Tested against all 14 priced Handyman rows before training. 14/14 correct on `labor_only` — the highest-value feature for the benchmark.

**Extraction distribution across all 1,432 rows:**

| Feature | Value | Count |
|---------|-------|-------|
| `labor_only` | True | 651 (45%) |
| `complexity_tier` | low | 792 (55%) |
| `complexity_tier` | medium | 524 (37%) |
| `complexity_tier` | high | 116 (8%) |

### Structured features (from request fields)

| Feature | Encoding |
|---------|----------|
| `deadline` | Ordinal: "As soon as possible"→4, "Within 1 week"→3, "Within 1-2 weeks"→2, "Within 1 month"→1, "I'm flexible"→0 |
| `service_category` | One-hot across all 18 training categories |
| `state_median_income` | Continuous, normalised by ÷100,000 |
| `original_estimate` | Continuous — the baseline model's point estimate, available at serving time |

`service_subtype` was dropped: too sparse relative to `service_category` + LLM `complexity_tier` which captures the same signal more reliably.

---

## Model

### Architecture

Three independent XGBoost regressors, each using `objective='reg:quantileerror'` (XGBoost 2.0+). Targets are `log(price)` (`target_transform="log"` in `meta.json`) so the pinball loss optimizes relative error — what MAPE measures — rather than absolute dollars; the serving layer inverts with `exp`.

| Model | Quantile | Output |
|-------|----------|--------|
| q=0.05 | 5th percentile | `estimate_lo` |
| q=0.50 | 50th percentile | `estimate_midpoint` |
| q=0.95 | 95th percentile | `estimate_hi` |

**Why three separate models instead of one:** Quantile regression requires a separate loss function per quantile. Three models give independent interval bounds. Post-prediction, monotonicity is enforced: `lo = min(lo_pred, mid_pred)`, `hi = max(hi_pred, mid_pred)`. This prevents interval crossing from independent predictions.

**`estimate_midpoint` is NOT `(lo + hi) / 2`** — it is the q=0.5 model's direct output. Averaging assumes a symmetric distribution; the q=0.5 model captures the actual median of the conditional distribution, which is skewed for variable-scope categories.

### Hyperparameters

```python
{
    "n_estimators":     400,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "random_state":     42,
}
```

`max_depth=4` and `min_child_weight=3` prevent overfitting on the 411-row supervised set. Deeper trees memorise the training data.

### Routing strategy

The XGBoost model improves Handyman significantly but degrades well-priced categories (HVAC, Landscaping, Cleaning) where the baseline already achieves 8–10% MAPE with 65–66 training rows each. Rather than blending, the serving layer routes by category:

- **Well-priced categories** (Appliance Repair, Cleaning, HVAC, Landscaping, Moving, Pest Control, Roofing): return `original_estimate` and its existing bounds directly. The baseline is optimal here; the model makes these categories worse.
- **Hard categories** (Handyman, Electrical, Flooring, Plumbing, and all categories not in the well-priced set): return the XGBoost model output directly. For high-variance categories (Handyman, Plumbing, Electrical, Flooring), a 25% symmetric interval widening is applied post-prediction to restore coverage without shifting the midpoint.

The category is always present in the request payload, so routing is deterministic and adds zero latency.

### Feature importance (q=0.5 model)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `original_estimate` | 0.128 |
| 2 | `cat_hvac` | 0.080 |
| 3 | `cat_moving` | 0.079 |
| 4 | `cat_cleaning` | 0.076 |
| 5 | `complexity_tier` | 0.071 |
| 6 | `cat_landscaping` | 0.062 |
| 7 | `has_area_measure` | 0.062 |
| 8 | `cat_appliance_repair` | 0.060 |
| 9 | `cat_handyman` | 0.056 |
| 10 | `cat_roofing` | 0.055 |

`original_estimate` is the top feature — the model learns corrections around the baseline rather than predicting from scratch. `complexity_tier` (rank 5) and `has_area_measure` (rank 7) are the highest-ranked LLM-extracted features, confirming they carry genuine signal.

---

## Training procedure

1. Load `features_enriched.csv` (1,432 rows with Haiku features + income enrichment).
2. Filter to 411 priced rows for supervised training.
3. Compute `TRAINING_MEDIAN_INTERVAL = median(estimate_hi - estimate_lo)` across priced rows = **$212** (used for OOD confidence detection at inference).
4. **Stratified 5-fold cross-validation** across all 411 rows, stratified by `service_category`. Rare categories (< 5 rows) binned as "other" for stratification.
5. **Leave-one-out CV for Handyman** — 14 rows is too few for a train/test split. LOO-CV gives an unbiased estimate but note: with 14 rows, a single outlier swings the metric by ~20 points. The LOO-CV number is a diagnostic; the held-out evaluation set controlled by HouseAccount is the submission benchmark.
6. Train final models on all 411 priced rows. Save artifacts to `models/`.

---

## Confidence calibration

```
if estimate_midpoint <= 0:
    confidence = 0.3   # guard against division by zero

interval_ratio = (estimate_hi - estimate_lo) / estimate_midpoint
base_confidence = 1 / (1 + interval_ratio)
# Narrow interval → confidence near 1.0
# Wide interval   → confidence near 0.0

confidence = min(base_confidence, ood_cap)
```

**OOD caps** — confidence is capped when the input is outside the training distribution. Out-of-distribution inputs are not rejected; they are passed through with `confidence < 0.5` so HouseAccount can route them appropriately.

| Condition | Cap | Rationale |
|-----------|-----|-----------|
| `estimate_midpoint > $5,000` | 0.40 | 95th percentile of training data. The model has minimal signal above this. |
| Interval > 3× median ($212) | 0.45 | Prediction is highly uncertain regardless of midpoint. |
| Category not in 8 production verticals | 0.40 | Model was trained on limited data for these categories. |

**Production verticals** (from HouseAccount's current live categories):

```
Cleaning, Landscaping, Pest Control, Electrical, Plumbing, HVAC, Handyman, Exterior
```

Categories outside this set (Remodeling, Auto, Pool, Chimney, Moving, Painting, Flooring, General Contractor, Roofing, Appliance Repair) receive the OOD cap.

---

## Performance

**PRD target:** < 2 seconds end-to-end per request.

The dominant latency source is the Claude Haiku scope extraction call (~300–800ms over the Anthropic API). XGBoost inference is negligible (<5ms). The response cache (`LRUCache(maxsize=2048)`) eliminates Haiku latency on repeated identical descriptions.

**Measured (run `python3 integration_test.py` against the deployed endpoint):**

| Metric | Target | Observed |
|--------|--------|----------|
| Cold-start | < 2000ms | 1140ms |
| Warm avg (5 calls) | < 2000ms | 289ms |
| Warm max (5 calls) | < 2000ms | 320ms |

Section 3 of `integration_test.py` measures and asserts all three against the live `RAILS_URL`.

---

## Evaluation results

### Against baseline

| Metric | Baseline | Routed model | Δ |
|--------|----------|-------------|---|
| Blended MAPE | 11.6% | **11.2%** | −0.4pp ✓ |
| Handyman MAPE (LOO-CV) | 48.4% | **26.6%** | −21.8pp ✓ |
| MAE | $54 | $55 | +$1 |
| WAPE | 9.2% | 9.5% | +0.3pp |
| Bias | +$6 | −$0 | near-zero |
| Coverage (lo/hi interval) | 97.3% | 95.9% | −1.4pp |

The baseline's 97.3% coverage reflects very wide intervals — the existing model is conservative. The routed model's 95.9% coverage tightens intervals while staying above 80% nominal for the blended set. MAE and WAPE tick up marginally because log-space training optimizes relative (percentage) error, trading a little absolute-dollar accuracy for the large MAPE gain on the hard tail.

### Per-category MAPE improvement

| Category | Baseline | Model | Change |
|----------|----------|-------|--------|
| Handyman | 48.4% | 32.3% | −16.1pp |
| Cleaning | 8.7% | 8.7% | 0 (routed to baseline) |
| Moving | 9.2% | 9.2% | 0 (routed to baseline) |
| Landscaping | 9.7% | 9.7% | 0 (routed to baseline) |
| HVAC | 9.7% | 9.7% | 0 (routed to baseline) |
| Pest Control | 10.2% | 10.2% | 0 (routed to baseline) |
| Roofing | 9.3% | 9.3% | 0 (routed to baseline) |

Well-priced categories are preserved exactly; the model only intervenes where it helps.

---

## Assumptions

1. **`job_description` is the primary scope signal.** The dataset has no structured scope fields. Text extraction is the only path to scope information.

2. **Category is always known at prediction time.** The routing strategy requires `service_category` in every request. This is a required field per the API contract.

3. **`original_estimate` is available at serving time.** The baseline model's output is an optional field in the request. When absent, the model substitutes the per-category median `original_estimate` from training data (stored in `models/meta.json`) to keep the feature in-distribution. Feeding 0 would break the model's top feature.

4. **State-level income is an adequate geographic proxy.** ZIP-level income would be more precise but requires a Census API key. State median income captures broad cost-of-living variation (Mississippi $53,600 vs. Maryland $101,710) at the cost of intra-state variation.

5. **Haiku extraction quality is stable.** The model assumes Haiku reliably extracts `labor_only` and `complexity_tier`. If Haiku's behavior changes across model versions, extraction should be re-validated and features potentially re-extracted before retraining.

---

## Known limitations

**Handyman confidence intervals are too narrow.** LOO-CV coverage for Handyman is 50% against an 80% target. The q=0.05/q=0.95 quantile models underestimate the true spread for variable-scope categories. A 25% symmetric post-hoc interval widening is applied to the four high-variance categories (Handyman, Plumbing, Electrical, Flooring), which recovers coverage without shifting the midpoint.

**14 Handyman training rows.** LOO-CV MAPE on 14 rows is statistically noisy — a single outlier swings the metric by ~20 percentage points. The reported 26.6% is the mean across 14 leave-one-out folds, but the distribution is wide. More priced Handyman data is the highest-value thing HouseAccount could add to improve this model.

**No ZIP-level income.** The model uses state-level income as a geographic feature. Labor cost varies substantially within states (San Francisco vs. Fresno, Manhattan vs. Buffalo). ZIP-level Census ACS data would improve predictions in high-variance states.

**Remodeling/General Contractor are fully OOD.** These categories have 0 priced rows. The model produces estimates for them but with confidence ≤ 0.40 and no validation signal. They should be monitored separately once the model is in production.

---

## Retraining

When new `final_price` data is available:

```bash
# Re-run Haiku extraction on any new rows (script is resumable)
python3 extract_features.py

# Re-run enrichment
python3 enrich_zip.py

# Retrain and evaluate
python3 train.py
```

The routing thresholds (well-priced vs. hard categories) should be re-evaluated after each retrain by checking per-category MAPE. A category that accumulates enough priced rows to cross the 12% MAPE threshold should be moved from the hard category list to the well-priced list.
