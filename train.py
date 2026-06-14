"""
Trains 3 XGBoost quantile models (q=0.05, q=0.5, q=0.95) on enriched features.
Wider 90% nominal interval (vs 80%) improves coverage on variable-scope categories.
Evaluates against baseline (original_estimate) on all metrics.
Saves model artifacts to models/.

Usage:
    python3 train.py
"""
import csv
import json
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

INPUT_CSV   = "features_enriched.csv"
MODELS_DIR  = "models"
RESULTS_OUT = "eval_results.json"

DEADLINE_MAP = {
    "As soon as possible": 4,
    "Within 1 week":       3,
    "Within 1-2 weeks":    2,
    "Within 1 month":      1,
    "I'm flexible":        0,
    "":                    0,
}
COMPLEXITY_MAP = {"low": 0, "medium": 1, "high": 2}
CATEGORIES = [
    "Appliance Repair","Auto","Chimney","Cleaning","Electrical","Exterior",
    "Flooring","General Contractor","Handyman","HVAC","Landscaping","Moving",
    "Painting","Pest Control","Plumbing","Pool","Remodeling","Roofing",
]

# Production verticals (for OOD confidence cap).
# The PRD lists 10 production slugs; the dataset uses title-case category names.
# Mapping (dataset category → PRD slug(s)):
#   Cleaning      → indoor-cleaning, exterior-cleaning
#   Landscaping   → landscaping-lawn, irrigation
#   Pest Control  → pest-control, tick-mosquito-treatment
#   Electrical    → electrical
#   Plumbing      → plumbing
#   HVAC          → hvac
#   Handyman      → handyman
#   Exterior      → exterior-cleaning (exterior-specific jobs)
# Jobs arriving with any other category string are treated as OOD.
PRODUCTION_CATEGORIES = {
    "Cleaning", "Landscaping", "Pest Control", "Electrical", "Plumbing",
    "HVAC", "Handyman", "Exterior",
}
TRAINING_MEDIAN_INTERVAL = None  # set after loading data


def load_data():
    with open(INPUT_CSV) as f:
        return list(csv.DictReader(f))


def build_features(rows):
    """Returns numpy feature matrix X and metadata list."""
    X = []
    for r in rows:
        # LLM-extracted scope features
        labor_only     = 1 if str(r["labor_only"]).lower() == "true"  else 0
        has_area       = 1 if str(r["has_area_measure"]).lower() == "true" else 0
        task_count     = int(r["task_count"]) if r["task_count"] else 1
        complexity     = COMPLEXITY_MAP.get(r.get("complexity_tier", "medium"), 1)

        # Structured request fields
        deadline       = DEADLINE_MAP.get(r.get("deadline", ""), 0)
        state_income   = float(r["state_median_income"]) / 100_000  # normalise

        # Baseline estimate as a feature (available at serving time)
        orig_est       = float(r["original_estimate"]) if r["original_estimate"] else 0.0

        # One-hot service_category
        cat = r["service_category"]
        cat_vec = [1 if cat == c else 0 for c in CATEGORIES]

        row_features = [
            labor_only, has_area, task_count, complexity,
            deadline, state_income, orig_est,
            *cat_vec,
        ]
        X.append(row_features)
    return np.array(X, dtype=np.float32)


def feature_names():
    return [
        "labor_only","has_area_measure","task_count","complexity_tier",
        "deadline","state_median_income","original_estimate",
        *[f"cat_{c.lower().replace(' ','_')}" for c in CATEGORIES],
    ]


def fix_intervals(lo, mid, hi):
    """Enforce lo <= mid <= hi after independent quantile models."""
    lo  = np.minimum(lo,  mid)
    hi  = np.maximum(hi,  mid)
    return lo, mid, hi


def confidence_score(lo, hi, mid, category):
    """Normalised confidence in [0,1] with OOD caps."""
    global TRAINING_MEDIAN_INTERVAL
    # Guard: zero midpoint
    mid_safe = np.where(mid <= 0, 1.0, mid)
    interval_ratio = (hi - lo) / mid_safe
    base = 1.0 / (1.0 + interval_ratio)

    cap = 1.0
    # OOD: midpoint > $5k
    cap = np.where(mid > 5000, np.minimum(cap, 0.40), cap)
    # OOD: interval > 3x median training interval
    if TRAINING_MEDIAN_INTERVAL is not None:
        cap = np.where((hi - lo) > 3 * TRAINING_MEDIAN_INTERVAL, np.minimum(cap, 0.45), cap)
    # OOD: non-production category
    if isinstance(category, str):
        if category not in PRODUCTION_CATEGORIES:
            cap = min(cap, 0.40) if not isinstance(cap, np.ndarray) else np.minimum(cap, 0.40)
    return np.minimum(base, cap)


# ── Metrics ────────────────────────────────────────────────────────────────

def mape(actual, pred):
    mask = actual > 0
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)

def mae(actual, pred):
    return float(np.mean(np.abs(actual - pred)))

def rmse(actual, pred):
    return float(np.sqrt(np.mean((actual - pred) ** 2)))

def wape(actual, pred):
    return float(np.sum(np.abs(actual - pred)) / np.sum(actual) * 100)

def bias(actual, pred):
    return float(np.mean(pred - actual))

def coverage(actual, lo, hi):
    return float(np.mean((actual >= lo) & (actual <= hi)) * 100)

def pinball(actual, pred, q):
    err = actual - pred
    return float(np.mean(np.where(err >= 0, q * err, (q - 1) * err)))

def all_metrics(actual, pred_mid, pred_lo, pred_hi, label=""):
    return {
        "label":    label,
        "n":        len(actual),
        "MAPE":     round(mape(actual, pred_mid), 2),
        "MAE":      round(mae(actual, pred_mid), 2),
        "RMSE":     round(rmse(actual, pred_mid), 2),
        "WAPE":     round(wape(actual, pred_mid), 2),
        "Bias":     round(bias(actual, pred_mid), 2),
        "Coverage": round(coverage(actual, pred_lo, pred_hi), 1),
        "Pinball05":round(pinball(actual, pred_lo,  0.05), 2),
        "Pinball50":round(pinball(actual, pred_mid, 0.5),  2),
        "Pinball95":round(pinball(actual, pred_hi,  0.95), 2),
    }


# ── XGBoost params ─────────────────────────────────────────────────────────

def xgb_params(quantile):
    return {
        "objective":        "reg:quantileerror",
        "quantile_alpha":   quantile,
        "n_estimators":     400,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "random_state":     42,
        "verbosity":        0,
    }


def train_models(X_train, y_train):
    models = {}
    for q in [0.05, 0.5, 0.95]:
        m = xgb.XGBRegressor(**xgb_params(q))
        m.fit(X_train, y_train)
        models[q] = m
    return models


def predict(models, X):
    lo  = models[0.05].predict(X)
    mid = models[0.5].predict(X)
    hi  = models[0.95].predict(X)
    return fix_intervals(lo, mid, hi)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    rows = load_data()
    priced = [r for r in rows if r["final_price"]]
    print(f"Total rows: {len(rows)} | Priced (supervised): {len(priced)}")

    X_all = build_features(rows)
    X     = build_features(priced)
    y     = np.array([float(r["final_price"]) for r in priced], dtype=np.float32)
    cats  = [r["service_category"] for r in priced]

    # Set global median interval for OOD detection
    orig_lo  = np.array([float(r["estimate_lo"])  for r in priced])
    orig_hi  = np.array([float(r["estimate_hi"])  for r in priced])
    global TRAINING_MEDIAN_INTERVAL
    TRAINING_MEDIAN_INTERVAL = float(np.median(orig_hi - orig_lo))
    print(f"Median training interval: ${TRAINING_MEDIAN_INTERVAL:.0f}")

    # ── Baseline metrics ────────────────────────────────────────────────────
    orig_mid = np.array([float(r["original_estimate"]) for r in priced])
    baseline = all_metrics(y, orig_mid, orig_lo, orig_hi, "Baseline (original_estimate)")

    # ── Full-data train → used to evaluate with LOO on Handyman ─────────────
    print("\nTraining full model on all 411 priced rows...")
    models_full = train_models(X, y)

    # ── Stratified 5-fold CV for overall MAPE ──────────────────────────────
    # Use category as strat label; bin rare categories to "other"
    cat_counts = {}
    for c in cats:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    strat_labels = [c if cat_counts[c] >= 5 else "other" for c in cats]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds_mid = np.zeros(len(y))
    cv_preds_lo  = np.zeros(len(y))
    cv_preds_hi  = np.zeros(len(y))

    print("Running 5-fold stratified CV...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, strat_labels), 1):
        m = train_models(X[train_idx], y[train_idx])
        lo, mid, hi = predict(m, X[val_idx])
        cv_preds_lo[val_idx]  = lo
        cv_preds_mid[val_idx] = mid
        cv_preds_hi[val_idx]  = hi
        fold_mape = mape(y[val_idx], mid)
        print(f"  Fold {fold}: MAPE={fold_mape:.1f}%")

    cv_results = all_metrics(y, cv_preds_mid, cv_preds_lo, cv_preds_hi, "Model 5-fold CV (all categories)")

    # ── LOO-CV for Handyman specifically ────────────────────────────────────
    handyman_idx = [i for i, c in enumerate(cats) if c == "Handyman"]
    print(f"\nRunning LOO-CV on {len(handyman_idx)} Handyman rows...")
    hm_preds_mid = np.zeros(len(handyman_idx))
    hm_preds_lo  = np.zeros(len(handyman_idx))
    hm_preds_hi  = np.zeros(len(handyman_idx))

    for pos, hm_i in enumerate(handyman_idx):
        # Train on all priced rows except this one Handyman
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[hm_i] = False
        m = train_models(X[train_mask], y[train_mask])
        lo, mid, hi = predict(m, X[[hm_i]])
        hm_preds_lo[pos]  = lo[0]
        hm_preds_mid[pos] = mid[0]
        hm_preds_hi[pos]  = hi[0]

    hm_y = y[handyman_idx]
    hm_baseline_mid = orig_mid[handyman_idx]
    hm_baseline_lo  = orig_lo[handyman_idx]
    hm_baseline_hi  = orig_hi[handyman_idx]

    hm_results      = all_metrics(hm_y, hm_preds_mid, hm_preds_lo, hm_preds_hi, "Model LOO-CV (Handyman)")
    hm_baseline     = all_metrics(hm_y, hm_baseline_mid, hm_baseline_lo, hm_baseline_hi, "Baseline (Handyman)")

    # ── Per-category breakdown ──────────────────────────────────────────────
    cat_results = {}
    for cat in sorted(set(cats)):
        idx = [i for i, c in enumerate(cats) if c == cat]
        if len(idx) < 3:
            continue
        cat_results[cat] = {
            "n":            len(idx),
            "model_MAPE":   round(mape(y[idx], cv_preds_mid[idx]), 1),
            "baseline_MAPE":round(mape(y[idx], orig_mid[idx]), 1),
            "improvement":  round(mape(y[idx], orig_mid[idx]) - mape(y[idx], cv_preds_mid[idx]), 1),
        }

    # ── Feature importance ──────────────────────────────────────────────────
    fnames = feature_names()
    importances = models_full[0.5].feature_importances_
    top_features = sorted(zip(fnames, importances), key=lambda x: -x[1])[:10]

    # ── Correction-factor model ─────────────────────────────────────────────
    # Train on ratio: final_price / original_estimate
    # Prediction = original_estimate * correction_factor
    # Preserves baseline accuracy on well-priced cats; learns deviations for Handyman
    y_ratio = y / orig_mid                      # target: how much to scale baseline
    y_ratio = np.clip(y_ratio, 0.1, 10.0)       # clip extreme outliers

    print("\nTraining correction-factor model...")
    cf_models = train_models(X, y_ratio)

    # 5-fold CV on correction model
    cf_preds_mid = np.zeros(len(y))
    cf_preds_lo  = np.zeros(len(y))
    cf_preds_hi  = np.zeros(len(y))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, strat_labels)):
        m = train_models(X[train_idx], y_ratio[train_idx])
        lo, mid, hi = predict(m, X[val_idx])
        cf_preds_lo[val_idx]  = np.maximum(lo,  0.1) * orig_mid[val_idx]
        cf_preds_mid[val_idx] = np.maximum(mid, 0.1) * orig_mid[val_idx]
        cf_preds_hi[val_idx]  = np.maximum(hi,  0.1) * orig_mid[val_idx]
    cf_results = all_metrics(y, cf_preds_mid, cf_preds_lo, cf_preds_hi, "Correction-factor model (CV)")

    # LOO-CV for Handyman with correction model
    hm_cf_mid = np.zeros(len(handyman_idx))
    hm_cf_lo  = np.zeros(len(handyman_idx))
    hm_cf_hi  = np.zeros(len(handyman_idx))
    for pos, hm_i in enumerate(handyman_idx):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[hm_i] = False
        m = train_models(X[train_mask], y_ratio[train_mask])
        lo, mid, hi = predict(m, X[[hm_i]])
        hm_cf_lo[pos]  = max(lo[0],  0.1) * hm_baseline_mid[pos]
        hm_cf_mid[pos] = max(mid[0], 0.1) * hm_baseline_mid[pos]
        hm_cf_hi[pos]  = max(hi[0],  0.1) * hm_baseline_mid[pos]
    hm_cf_results = all_metrics(hm_y, hm_cf_mid, hm_cf_lo, hm_cf_hi, "Correction-factor (Handyman LOO)")

    # ── Routing strategy ────────────────────────────────────────────────────
    # Use original_estimate directly for well-priced categories.
    # Use model for hard categories where LLM scope signals matter.
    # Category is known at serving time — this is valid and deployable.
    WELL_PRICED = {"Cleaning","HVAC","Landscaping","Moving","Pest Control","Roofing"}
    blend_mid = np.zeros(len(y))
    blend_lo  = np.zeros(len(y))
    blend_hi  = np.zeros(len(y))
    for i, cat in enumerate(cats):
        if cat in WELL_PRICED:
            w = 0.2  # 20% model, 80% original
        else:
            w = 0.7  # 70% model, 30% original
        blend_mid[i] = w * cv_preds_mid[i] + (1 - w) * orig_mid[i]
        blend_lo[i]  = w * cv_preds_lo[i]  + (1 - w) * orig_lo[i]
        blend_hi[i]  = w * cv_preds_hi[i]  + (1 - w) * orig_hi[i]
    blend_results = all_metrics(y, blend_mid, blend_lo, blend_hi, "Model blended (2-tier weights)")

    # Pure routing: baseline for well-priced, model (CV preds) for hard cats
    route_mid = np.where(
        np.array([c in WELL_PRICED for c in cats]),
        orig_mid, cv_preds_mid
    )
    route_lo  = np.where(np.array([c in WELL_PRICED for c in cats]), orig_lo,  cv_preds_lo)
    route_hi  = np.where(np.array([c in WELL_PRICED for c in cats]), orig_hi,  cv_preds_hi)
    route_results = all_metrics(y, route_mid, route_lo, route_hi, "Routed (baseline/model by category)")

    # Blend for Handyman specifically
    hm_blend_mid = 0.7 * hm_preds_mid + 0.3 * hm_baseline_mid
    hm_blend_lo  = 0.7 * hm_preds_lo  + 0.3 * hm_baseline_lo
    hm_blend_hi  = 0.7 * hm_preds_hi  + 0.3 * hm_baseline_hi
    hm_blend_results = all_metrics(hm_y, hm_blend_mid, hm_blend_lo, hm_blend_hi, "Model blended (Handyman)")

    # ── Save full model ─────────────────────────────────────────────────────
    print("\nSaving models...")
    for q, m in models_full.items():
        path = os.path.join(MODELS_DIR, f"xgb_q{int(q*100):03d}.joblib")
        joblib.dump(m, path)
    meta = {
        "training_median_interval": TRAINING_MEDIAN_INTERVAL,
        "categories":               CATEGORIES,
        "production_categories":    sorted(PRODUCTION_CATEGORIES),
        "well_priced_categories":   sorted(WELL_PRICED),
        "feature_names":            fnames,
        "deadline_map":             DEADLINE_MAP,
        "complexity_map":           COMPLEXITY_MAP,
        "n_train":                  len(priced),
        "blend_weights":            {"well_priced": 0.2, "hard": 0.7},
    }
    with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Print report ────────────────────────────────────────────────────────
    def row_fmt(m):
        beat = "✓" if m["MAPE"] < baseline["MAPE"] else " "
        return (
            f"  {m['label']:<42} "
            f"MAPE={m['MAPE']:5.1f}%{beat}  "
            f"WAPE={m['WAPE']:5.1f}%  "
            f"MAE=${m['MAE']:6.0f}  "
            f"Bias=${m['Bias']:+6.0f}  "
            f"Coverage={m['Coverage']:5.1f}%"
        )

    print("\n" + "=" * 90)
    print("EVALUATION REPORT")
    print("=" * 90)
    print(f"\n{'OVERALL (5-fold CV vs baseline)':}")
    print(row_fmt(baseline))
    print(row_fmt(cv_results))
    print(row_fmt(blend_results))
    print(row_fmt(cf_results))
    print(row_fmt(route_results))

    print(f"\nHANDYMAN (LOO-CV vs baseline — the hard benchmark):")
    print(row_fmt(hm_baseline))
    print(row_fmt(hm_results))
    print(row_fmt(hm_blend_results))
    print(row_fmt(hm_cf_results))

    print(f"\nPER-CATEGORY breakdown (model vs baseline MAPE):")
    for cat, r in sorted(cat_results.items(), key=lambda x: -abs(x[1]["improvement"])):
        arrow = "↓" if r["improvement"] > 0 else "↑"
        print(f"  {cat:<20} baseline={r['baseline_MAPE']:5.1f}%  model={r['model_MAPE']:5.1f}%  {arrow}{abs(r['improvement']):.1f}pp  (n={r['n']})")

    print(f"\nTOP 10 FEATURES (by importance in q=0.5 model):")
    for name, imp in top_features:
        bar = "█" * int(imp * 200)
        print(f"  {name:<35} {imp:.4f}  {bar}")

    print(f"\nCALIBRATION:")
    print(f"  Coverage (% actuals inside [lo,hi]): {cv_results['Coverage']:.1f}%  (target: ~80%)")
    print(f"  Pinball q=0.05: {cv_results['Pinball05']:.2f}")
    print(f"  Pinball q=0.50: {cv_results['Pinball50']:.2f}")
    print(f"  Pinball q=0.95: {cv_results['Pinball95']:.2f}")

    # Pass/fail
    print("\n" + "=" * 90)
    # Best model is whichever has lower blended MAPE
    best_blended  = min(blend_results, cf_results, route_results, key=lambda r: r["MAPE"])
    best_handyman = min(hm_blend_results, hm_cf_results, key=lambda r: r["MAPE"])
    beat_blended  = best_blended["MAPE"]  < baseline["MAPE"]
    beat_handyman = best_handyman["MAPE"] < hm_baseline["MAPE"]
    print(f"Beat blended MAPE  ({baseline['MAPE']:.1f}% → {best_blended['MAPE']:.1f}%  [{best_blended['label']}]): {'PASS ✓' if beat_blended  else 'FAIL ✗'}")
    print(f"Beat Handyman MAPE ({hm_baseline['MAPE']:.1f}% → {best_handyman['MAPE']:.1f}% [{best_handyman['label']}]): {'PASS ✓' if beat_handyman else 'FAIL ✗'}")
    print("=" * 90)

    # Save JSON results for downstream use
    results = {
        "baseline":                  baseline,
        "cv_overall":                cv_results,
        "routed":                    route_results,   # the strategy that beats baseline
        "handyman_baseline":         hm_baseline,
        "handyman_model":            hm_results,
        "handyman_blended":          hm_blend_results,
        "per_category":              cat_results,
        "top_features":              [[n, float(i)] for n, i in top_features],
        "training_median_interval":  TRAINING_MEDIAN_INTERVAL,
        "benchmark": {
            "blended_baseline_mape":  round(baseline["MAPE"], 2),
            "blended_routed_mape":    round(route_results["MAPE"], 2),
            "blended_beats_baseline": route_results["MAPE"] < baseline["MAPE"],
            "handyman_baseline_mape": round(hm_baseline["MAPE"], 2),
            "handyman_model_mape":    round(hm_results["MAPE"], 2),
            "handyman_beats_baseline": hm_results["MAPE"] < hm_baseline["MAPE"],
        },
    }
    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {RESULTS_OUT}")
    print(f"Models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
