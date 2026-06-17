"""
Experiment: does Conformalized Quantile Regression (CQR) fix the interval
coverage gap without blowing up interval width?

The model's [q05, q95] band is nominally 90% but under-covers in CV (~70%).
CQR calibrates a finite-sample offset Q on a held-out calibration split so the
band becomes [q05 - Q, q95 + Q], which carries a coverage guarantee on
exchangeable data. This measures raw vs CQR coverage and median width — it does
NOT touch production serving.

Honest protocol: outer 5-fold CV. Each fold's training rows are split into
proper-train (fit quantile models) and calibration (compute Q). The offset is
applied to the held-out test fold only, so reported coverage isn't inflated.

Run: python experiment_conformal.py
"""
import json
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import KFold

import train as T

ALPHA = 0.10          # target 90% coverage ([q05, q95] nominal)
TARGET = 1 - ALPHA
CALIB_FRAC = 0.30     # share of each train fold held out to calibrate Q
SEED = 42
RESULTS_OUT = "experiment_conformal_results.json"


def load():
    rows = [r for r in T.load_data() if r["final_price"]]
    X = T.build_features(rows)
    y = np.array([float(r["final_price"]) for r in rows], dtype=np.float32)
    return X, y


def run():
    X, y = load()
    n = len(y)
    raw_lo = np.zeros(n); raw_hi = np.zeros(n)
    cqr_lo = np.zeros(n); cqr_hi = np.zeros(n)
    offsets = []

    print(f"CQR experiment: target {TARGET:.0%} coverage, {n} priced jobs, nested 5-fold CV")
    outer = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(outer.split(X), 1):
        # split train fold -> proper-train + calibration
        rng = np.random.RandomState(SEED + fold)
        perm = rng.permutation(tr)
        n_cal = max(20, int(len(tr) * CALIB_FRAC))
        cal, pro = perm[:n_cal], perm[n_cal:]

        models = T.train_models(X[pro], y[pro])
        lo_cal, _, hi_cal = T.predict(models, X[cal])
        # conformity score: how far the truth falls outside [lo, hi]
        scores = np.maximum(lo_cal - y[cal], y[cal] - hi_cal)
        # finite-sample conformal quantile level
        q_level = min(1.0, math.ceil((n_cal + 1) * TARGET) / n_cal)
        Q = float(np.quantile(scores, q_level, method="higher"))
        offsets.append(Q)

        lo_te, _, hi_te = T.predict(models, X[te])
        raw_lo[te], raw_hi[te] = lo_te, hi_te
        cqr_lo[te], cqr_hi[te] = lo_te - Q, hi_te + Q
        print(f"  Fold {fold}: Q=${Q:.0f}  "
              f"raw cov={T.coverage(y[te], lo_te, hi_te):.1f}%  "
              f"cqr cov={T.coverage(y[te], lo_te - Q, hi_te + Q):.1f}%")

    raw_cov = round(T.coverage(y, raw_lo, raw_hi), 1)
    cqr_cov = round(T.coverage(y, cqr_lo, cqr_hi), 1)
    raw_w = round(float(np.median(raw_hi - raw_lo)), 0)
    cqr_w = round(float(np.median(cqr_hi - cqr_lo)), 0)

    print("\n" + "=" * 70)
    print("CONFORMAL INTERVAL EXPERIMENT")
    print("=" * 70)
    print(f"  target coverage:        {TARGET:.0%}")
    print(f"  RAW   [q05,q95]  coverage={raw_cov:5.1f}%   median width=${raw_w:.0f}")
    print(f"  CQR   conformalized coverage={cqr_cov:5.1f}%   median width=${cqr_w:.0f}")
    print(f"  mean offset Q applied: ${np.mean(offsets):.0f}")
    hit = abs(cqr_cov - TARGET * 100) <= 5
    print(f"\n  CQR reaches target ({TARGET:.0%} +/- 5pp): {'YES' if hit else 'NO'}")
    print(f"  width cost: {raw_w:.0f} -> {cqr_w:.0f} (+{(cqr_w/raw_w - 1)*100:.0f}%)")
    print("=" * 70)

    with open(RESULTS_OUT, "w") as f:
        json.dump({"target": TARGET, "raw": {"coverage": raw_cov, "median_width": raw_w},
                   "cqr": {"coverage": cqr_cov, "median_width": cqr_w},
                   "mean_offset": round(float(np.mean(offsets)), 1),
                   "reaches_target": bool(hit)}, f, indent=2)
    print(f"\nSaved to {RESULTS_OUT}")


if __name__ == "__main__":
    run()
