"""
Experiment: learned per-category routing vs the hand-tuned routing in train.py.

BEFORE (production): hardcoded WELL_PRICED set + fixed blend weights
  (0.2 model for well-priced categories, 0.7 for hard ones), blending the
  correction-factor model with the baseline (original_estimate).
AFTER (experiment): per-category blend weight chosen from a grid, where w=0 is
  pure baseline and w=1 is pure correction-factor. Selection minimizes MAPE.

Honest evaluation (no selection-bias inflation): nested CV. The per-category
weight is chosen on INNER folds of the outer-training data only, then applied to
the held-out OUTER fold. BEFORE and AFTER score the same outer-fold CF
predictions, so the only difference is fixed vs learned weights. Categories with
fewer than MIN_ROWS rows default to baseline (w=0), which can never regress.

Run: python experiment_routing.py
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold

import train as T

WEIGHTS    = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]   # 0 = baseline, 1 = pure CF
MIN_ROWS   = 3        # categories thinner than this default to baseline
OUTER_SEED = 42
WELL_PRICED = {  # the production hand-tuned set (train.py:347)
    "Appliance Repair", "Cleaning", "HVAC", "Landscaping",
    "Moving", "Pest Control", "Roofing",
}
RESULTS_OUT = "experiment_routing_results.json"


def load():
    rows   = T.load_data()
    priced = [r for r in rows if r["final_price"]]
    X      = T.build_features(priced)
    y      = np.array([float(r["final_price"]) for r in priced], dtype=np.float32)
    orig   = np.array([float(r["original_estimate"]) for r in priced], dtype=np.float32)
    cats   = np.array([r["service_category"] for r in priced])
    return X, y, orig, cats


def cf_price(X_tr, y_tr, orig_tr, X_te, orig_te):
    """Correction-factor point model: predict final/original ratio, rescale."""
    ratio = np.clip(y_tr / orig_tr, 0.1, 10.0)
    m = xgb.XGBRegressor(**T.xgb_params(0.5))   # median only; intervals not needed here
    m.fit(X_tr, ratio)
    return np.maximum(m.predict(X_te), 0.1) * orig_te


def oof_cf(X, y, orig, idx, seed):
    """Out-of-fold CF price preds for the rows in idx (global-aligned output)."""
    idx  = np.asarray(idx)
    out  = np.zeros(len(idx))
    k    = min(5, len(idx))
    if k < 2:
        out[:] = cf_price(X[idx], y[idx], orig[idx], X[idx], orig[idx])
        return out
    for tr, te in KFold(n_splits=k, shuffle=True, random_state=seed).split(idx):
        out[te] = cf_price(X[idx[tr]], y[idx[tr]], orig[idx[tr]], X[idx[te]], orig[idx[te]])
    return out


def choose_policy(y, orig, cats, cf_inner, train_idx):
    """Per-category best blend weight on inner OOF preds; thin cats -> baseline."""
    tr = np.asarray(train_idx)
    policy = {}
    for cat in np.unique(cats[tr]):
        c = tr[cats[tr] == cat]
        if len(c) < MIN_ROWS:
            policy[cat] = 0.0
            continue
        best_w, best = 0.0, np.inf
        for w in WEIGHTS:
            pred = w * cf_inner[c] + (1 - w) * orig[c]
            mp = T.mape(y[c], pred)
            if mp < best:
                best, best_w = mp, w
        policy[cat] = best_w
    return policy


def metrics(y, pred):
    return {"MAPE": round(T.mape(y, pred), 2),
            "WAPE": round(T.wape(y, pred), 2),
            "MAE":  round(T.mae(y, pred), 2)}


def run():
    X, y, orig, cats = load()
    n = len(y)

    counts = {c: int((cats == c).sum()) for c in np.unique(cats)}
    strat  = np.array([c if counts[c] >= 5 else "other" for c in cats])
    w_fixed = lambda cat: 0.2 if cat in WELL_PRICED else 0.7

    before = np.zeros(n)
    after  = np.zeros(n)
    fold_policies = []

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    print("Running nested CV (5 outer folds, 5 inner folds each)...")
    for fold, (tr, te) in enumerate(outer.split(X, strat), 1):
        cf_te = cf_price(X[tr], y[tr], orig[tr], X[te], orig[te])     # shared by before/after
        wb = np.array([w_fixed(c) for c in cats[te]])
        before[te] = wb * cf_te + (1 - wb) * orig[te]

        cf_inner = np.zeros(n)
        cf_inner[tr] = oof_cf(X, y, orig, tr, seed=100 + fold)
        policy = choose_policy(y, orig, cats, cf_inner, tr)
        fold_policies.append(policy)
        wa = np.array([policy.get(c, 0.0) for c in cats[te]])
        after[te] = wa * cf_te + (1 - wa) * orig[te]
        print(f"  Fold {fold}: before MAPE={T.mape(y[te], before[te]):.1f}%  "
              f"after MAPE={T.mape(y[te], after[te]):.1f}%")

    baseline_o = metrics(y, orig)
    before_o   = metrics(y, before)
    after_o    = metrics(y, after)

    # Per-category MAPE
    per_cat = {}
    for cat in sorted(np.unique(cats)):
        m = cats == cat
        if m.sum() < MIN_ROWS:
            continue
        per_cat[cat] = {
            "n":        int(m.sum()),
            "baseline": round(T.mape(y[m], orig[m]),   1),
            "before":   round(T.mape(y[m], before[m]), 1),
            "after":    round(T.mape(y[m], after[m]),  1),
        }

    # Recommended deployable policy: select once on full-data inner OOF
    cf_full = np.zeros(n)
    cf_full[:] = oof_cf(X, y, orig, np.arange(n), seed=7)
    final_policy = choose_policy(y, orig, cats, cf_full, np.arange(n))

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("ROUTING EXPERIMENT — before (hand-tuned) vs after (learned per-category)")
    print("=" * 78)
    print(f"\nOVERALL ({n} priced jobs, nested 5-fold CV)")
    print(f"  {'Baseline (provider estimate)':<34} MAPE={baseline_o['MAPE']:5.1f}%  "
          f"WAPE={baseline_o['WAPE']:5.1f}%  MAE=${baseline_o['MAE']:.0f}")
    print(f"  {'BEFORE  hand-tuned routed CF':<34} MAPE={before_o['MAPE']:5.1f}%  "
          f"WAPE={before_o['WAPE']:5.1f}%  MAE=${before_o['MAE']:.0f}")
    print(f"  {'AFTER   learned per-category':<34} MAPE={after_o['MAPE']:5.1f}%  "
          f"WAPE={after_o['WAPE']:5.1f}%  MAE=${after_o['MAE']:.0f}")
    delta = round(before_o["MAPE"] - after_o["MAPE"], 2)
    print(f"\n  Δ MAPE (before → after): {delta:+.2f}pp"
          f"  {'(improvement)' if delta > 0 else '(no gain)' if delta == 0 else '(regression)'}")

    print(f"\nPER-CATEGORY MAPE (baseline / before / after)")
    print(f"  {'category':<20} {'n':>3}  {'base':>6} {'before':>7} {'after':>6}   note")
    regressions = []
    for cat, r in sorted(per_cat.items(), key=lambda x: -x[1]["n"]):
        note = ""
        if r["after"] > r["baseline"] + 0.5:
            note = "REGRESSED vs baseline"
            regressions.append(cat)
        elif r["after"] < r["before"] - 0.5:
            note = "improved vs before"
        print(f"  {cat:<20} {r['n']:>3}  {r['baseline']:>6.1f} {r['before']:>7.1f} "
              f"{r['after']:>6.1f}   {note}")

    print(f"\nRECOMMENDED PER-CATEGORY WEIGHT (0=baseline only, 1=pure CF model)")
    for cat in sorted(final_policy, key=lambda c: -counts.get(c, 0)):
        if counts.get(cat, 0) < MIN_ROWS:
            continue
        print(f"  {cat:<20} w={final_policy[cat]:.1f}   (hand-tuned was "
              f"{0.2 if cat in WELL_PRICED else 0.7})")

    print("\n" + "=" * 78)
    verdict = ("AFTER wins" if after_o["MAPE"] < before_o["MAPE"] and not regressions
               else "AFTER wins but regresses a category" if after_o["MAPE"] < before_o["MAPE"]
               else "no improvement")
    print(f"VERDICT: {verdict}. Regressions (>0.5pp vs baseline): "
          f"{regressions or 'none'}")
    print("=" * 78)

    with open(RESULTS_OUT, "w") as f:
        json.dump({"overall": {"baseline": baseline_o, "before": before_o, "after": after_o,
                               "delta_mape_pp": delta},
                   "per_category": per_cat,
                   "recommended_policy": final_policy,
                   "regressions": regressions}, f, indent=2)
    print(f"\nSaved to {RESULTS_OUT}")


if __name__ == "__main__":
    run()
