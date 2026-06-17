"""
Producer for VERIFIED outcomes: submit real final prices for completed jobs so
the model retrains as labels accumulate. This is the trusted side of the
feedback loop (public demo submissions are recorded as 'demo' and never train).

Each row must reference a job_id the pricing system already issued — the estimate
must exist in the ML store, or the outcome can't join its feature vector and
won't contribute to retraining.

Usage:
  GAUNTLET_PRICING_SECRET=... python submit_outcomes.py completed.csv \
      --base-url https://pricing-ml-production.up.railway.app
      # completed.csv has a header row: job_id,final_price

  python submit_outcomes.py completed.csv --dry-run    # print, send nothing
"""
import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_BASE = "http://localhost:8001"
OUTCOME_PATH = "/.netlify/functions/pricing-outcome"


def submit(base, secret, job_id, price):
    req = urllib.request.Request(
        base.rstrip("/") + OUTCOME_PATH,
        data=json.dumps({"job_id": job_id, "final_price": price,
                         "source": "verified"}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {secret}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.status, json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", help="CSV with columns: job_id,final_price")
    ap.add_argument("--base-url", default=os.environ.get("PRICING_ML_BASE", DEFAULT_BASE))
    ap.add_argument("--dry-run", action="store_true", help="print planned posts, send nothing")
    args = ap.parse_args()

    secret = os.environ.get("GAUNTLET_PRICING_SECRET", "")
    if not args.dry_run and not secret:
        sys.exit("GAUNTLET_PRICING_SECRET not set (needed to submit verified outcomes)")

    rows = list(csv.DictReader(open(args.csv_path)))
    sent = joined = failed = 0
    for r in rows:
        job_id = (r.get("job_id") or "").strip()
        try:
            price = float(r["final_price"])
        except (KeyError, ValueError, TypeError):
            print(f"skip (bad final_price): {r}"); failed += 1; continue
        if not job_id or price <= 0:
            print(f"skip (missing job_id or non-positive price): {r}"); failed += 1; continue

        if args.dry_run:
            print(f"DRY POST {args.base_url}{OUTCOME_PATH}  "
                  f"job_id={job_id} final_price={price} source=verified")
            continue
        try:
            status, body = submit(args.base_url, secret, job_id, price)
            sent += 1
            if body.get("ape") is not None:
                joined += 1
            print(f"ok   job_id={job_id} status={status} ape={body.get('ape')}")
        except urllib.error.HTTPError as e:
            failed += 1
            print(f"FAIL job_id={job_id} status={e.code} {e.read()[:160]}")
        except Exception as e:  # noqa: BLE001 — surface any transport error per row
            failed += 1
            print(f"FAIL job_id={job_id} {type(e).__name__}: {e}")

    print(f"\nsubmitted={sent} joined_estimate={joined} failed={failed} total={len(rows)}")
    if not args.dry_run and sent and joined == 0:
        print("WARNING: no outcomes joined a stored estimate — these job_ids have no "
              "estimate on record, so they will NOT contribute to retraining.")


if __name__ == "__main__":
    main()
