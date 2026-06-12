"""
Step 6 — Integration test against HouseAccount staging endpoint.

Tests the full round-trip:
  1. Posts a booking payload to YOUR Rails endpoint
  2. Verifies the response shape matches Appendix A
  3. Posts to HouseAccount staging (pro.houseparty.dev) with our price estimate
     and verifies their system accepts it

Required env vars:
  GAUNTLET_PRICING_SECRET     — bearer token for YOUR endpoint
  RAILS_URL                   — default http://localhost:3000

HouseAccount staging (already configured — no extra env vars needed):
  App-Name:    gauntlet
  Signing key: Dq+pSVNf7DxSzQxxvRSeh0mk32e+V3m+KCTv9boGstI=

Auth scheme: HMAC-SHA256 of "timestamp.payload_body" using the base64-decoded signing secret.
Headers: App-Name, App-Timestamp, App-Signature

Usage:
  GAUNTLET_PRICING_SECRET=my-secret-key python3 integration_test.py
"""
import base64
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# Load .env if present
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

RAILS_URL  = os.environ.get("RAILS_URL", "http://localhost:3000")
MY_SECRET  = os.environ.get("GAUNTLET_PRICING_SECRET", "")

# HouseAccount staging — credentials from Claudio
HA_BASE_URL     = "https://pro.houseparty.dev"
HA_APP_NAME     = "gauntlet"
# Key is the raw base64 string as bytes (NOT decoded) — confirmed by trial
HA_SIGNING_SECRET = "Dq+pSVNf7DxSzQxxvRSeh0mk32e+V3m+KCTv9boGstI=".encode()

ENDPOINT = f"{RAILS_URL}/.netlify/functions/pricing-estimate"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

results = []


def check(name, passed, detail=""):
    status = PASS if passed else FAIL
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    results.append((name, passed))


def post(url, body, headers=None):
    headers = headers or {}
    headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return None, {"error": str(e)}


def ha_signed_headers(payload_body: str) -> dict:
    """Build HMAC-SHA256 signed headers for HouseAccount staging."""
    timestamp = str(int(time.time()))
    message   = f"{timestamp}.{payload_body}"
    signature = hmac.new(
        HA_SIGNING_SECRET,
        message.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    return {
        "App-Name":      HA_APP_NAME,
        "App-Timestamp": timestamp,
        "App-Signature": signature,
        "Content-Type":  "application/json",
    }


def ha_post(path: str, body: dict):
    """POST to HouseAccount staging with HMAC auth."""
    payload_body = json.dumps(body, separators=(',', ':'))  # compact — required for signature
    headers = ha_signed_headers(payload_body)
    url = f"{HA_BASE_URL}{path}"
    req = urllib.request.Request(
        url, data=payload_body.encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return None, {"error": str(e)}


def auth_headers():
    return {"Authorization": f"Bearer {MY_SECRET}"}


# ── Test payloads ──────────────────────────────────────────────────────────

CASES = [
    {
        "name":    "Handyman labor-only (key benchmark)",
        "payload": {
            "job_id":            "integration-handyman-001",
            "service_category":  "Handyman",
            "zip_code":          "33484",
            "job_description":   "Install 3 exterior shutters (we supply all shutters and hardware)",
            "deadline":          "Within 1-2 weeks",
            "original_estimate": 750.0,
            "original_estimate_lo": 600.0,
            "original_estimate_hi": 900.0,
        },
        "expect_confidence_below": None,
        "expect_midpoint_below":   700.0,  # model should correct downward from $750
    },
    {
        "name":    "HVAC (well-priced, routes to baseline)",
        "payload": {
            "job_id":            "integration-hvac-001",
            "service_category":  "HVAC",
            "zip_code":          "78704",
            "job_description":   "AC unit not cooling, needs refrigerant recharge and inspection",
            "deadline":          "As soon as possible",
            "original_estimate": 350.0,
            "original_estimate_lo": 280.0,
            "original_estimate_hi": 420.0,
        },
        "expect_midpoint_exact": 350.0,  # should passthrough original_estimate unchanged
    },
    {
        "name":    "OOD — Remodeling (confidence must be < 0.5)",
        "payload": {
            "job_id":            "integration-ood-001",
            "service_category":  "Remodeling",
            "zip_code":          "90210",
            "job_description":   "Full kitchen remodel, 400 sq ft, new cabinets and countertops",
            "deadline":          "Within 1 month",
            "original_estimate": 45000.0,
        },
        "expect_confidence_below": 0.5,
    },
    {
        "name":    "OOD — price above $5k (confidence must be < 0.5)",
        "payload": {
            "job_id":            "integration-ood-002",
            "service_category":  "Roofing",
            "zip_code":          "60649",
            "job_description":   "Full roof replacement, 2,500 sq ft house, architectural shingles",
            "deadline":          "Within 1-2 weeks",
            "original_estimate": 12000.0,
        },
        "expect_confidence_below": 0.5,
    },
    {
        "name":    "Multi-trade complexity (high complexity_tier)",
        "payload": {
            "job_id":            "integration-complex-001",
            "service_category":  "Handyman",
            "zip_code":          "33311",
            "job_description":   "3-4 small mixed repairs: electrical outlet, plumbing drip, carpentry patch, interior painting touch-up",
            "deadline":          "I'm flexible",
            "original_estimate": 1125.0,
        },
    },
]


# ── Section 1: Auth and validation ─────────────────────────────────────────

print("\n── Section 1: Auth & validation ──────────────────────────")

status, body = post(ENDPOINT, CASES[0]["payload"])  # no auth header
check("Missing auth → 401", status == 401 and body.get("error") == "Unauthorized")

status, body = post(ENDPOINT, CASES[0]["payload"],
                    {"Authorization": "Bearer wrong-token"})
check("Wrong token → 401", status == 401)

status, body = post(ENDPOINT, {"job_id": "x", "service_category": "Plumbing"},
                    auth_headers())
check("Missing zip_code → 400 with field name", status == 400 and "zip_code" in body.get("error", ""))

status, body = post(ENDPOINT, {},  # empty body — missing job_id
                    auth_headers())
check("Missing job_id → 400", status == 400 and body.get("error") == "job_id required")


# ── Section 2: Prediction quality ─────────────────────────────────────────

print("\n── Section 2: Prediction quality ─────────────────────────")

for case in CASES:
    status, body = post(ENDPOINT, case["payload"], auth_headers())

    if status != 200:
        check(case["name"], False, f"HTTP {status}: {body}")
        continue

    # Shape checks
    required_fields = ["ok","job_id","estimate_lo","estimate_hi","estimate_midpoint","confidence","model_version"]
    missing = [f for f in required_fields if f not in body]
    if missing:
        check(case["name"], False, f"missing fields: {missing}")
        continue

    ok = True
    notes = []

    # ok=true
    if not body["ok"]:
        ok = False; notes.append("ok!=true")

    # job_id echoed
    if body["job_id"] != case["payload"]["job_id"]:
        ok = False; notes.append("job_id not echoed")

    # lo <= midpoint <= hi
    if not (body["estimate_lo"] <= body["estimate_midpoint"] <= body["estimate_hi"]):
        ok = False; notes.append(f"interval invalid: {body['estimate_lo']} <= {body['estimate_midpoint']} <= {body['estimate_hi']}")

    # confidence in [0, 1]
    if not (0.0 <= body["confidence"] <= 1.0):
        ok = False; notes.append(f"confidence out of range: {body['confidence']}")

    # OOD confidence cap
    if case.get("expect_confidence_below") and body["confidence"] >= case["expect_confidence_below"]:
        ok = False; notes.append(f"confidence {body['confidence']:.2f} >= {case['expect_confidence_below']} (OOD not capped)")

    # Midpoint correction check
    if case.get("expect_midpoint_below") and body["estimate_midpoint"] >= case["expect_midpoint_below"]:
        notes.append(f"midpoint ${body['estimate_midpoint']:.0f} >= ${case['expect_midpoint_below']:.0f} (model may not be correcting)")

    # Exact passthrough for well-priced categories
    if case.get("expect_midpoint_exact") and abs(body["estimate_midpoint"] - case["expect_midpoint_exact"]) > 0.01:
        ok = False; notes.append(f"expected passthrough ${case['expect_midpoint_exact']:.0f}, got ${body['estimate_midpoint']:.0f}")

    detail = f"mid=${body['estimate_midpoint']:.0f} conf={body['confidence']:.2f}"
    if notes:
        detail += " | " + "; ".join(notes)
    check(case["name"], ok, detail)


# ── Section 3: Response time ───────────────────────────────────────────────

print("\n── Section 3: Response time ───────────────────────────────")
import time

times = []
for _ in range(3):
    t0 = time.time()
    status, _ = post(ENDPOINT, CASES[0]["payload"], auth_headers())
    times.append(time.time() - t0)

avg_ms = sum(times) / len(times) * 1000
check(f"Avg response time < 2000ms (warm)", avg_ms < 2000, f"{avg_ms:.0f}ms avg over 3 calls")


# ── Section 4: HouseAccount staging ───────────────────────────────────────
# Auth: HMAC-SHA256("timestamp.payload") — App-Name/App-Timestamp/App-Signature headers
# Endpoint: POST /api/bookings
# Our estimate goes in estimate: {min: estimate_lo, max: estimate_hi}

print("\n── Section 4: HouseAccount staging (pro.houseparty.dev) ──")

# Test cases: each includes the booking fields HouseAccount requires
_RUN_ID = str(int(time.time()))[-6:]  # last 6 digits of timestamp — unique per run

STAGING_CASES = [
    {
        "name": "Handyman labor-only booking",
        "pricing_payload": {
            "job_id":            f"staging-handyman-{_RUN_ID}",
            "service_category":  "Handyman",
            "zip_code":          "33484",
            "job_description":   "Install 3 supplied exterior shutters (we supply all shutters)",
            "deadline":          "Within 1-2 weeks",
            "original_estimate": 750.0,
            "original_estimate_lo": 600.0,
            "original_estimate_hi": 900.0,
        },
        "booking_extras": {
            "name":    "Test Homeowner",
            "phone":   f"555555{_RUN_ID[:4]}",
            "comment": "Gauntlet integration test — Handyman labor-only",
        },
    },
    {
        "name": "HVAC booking (well-priced category)",
        "pricing_payload": {
            "job_id":            f"staging-hvac-{_RUN_ID}",
            "service_category":  "HVAC",
            "zip_code":          "78704",
            "job_description":   "AC not cooling, needs refrigerant recharge and inspection",
            "deadline":          "As soon as possible",
            "original_estimate": 350.0,
            "original_estimate_lo": 280.0,
            "original_estimate_hi": 420.0,
        },
        "booking_extras": {
            "name":    "Test Homeowner",
            "phone":   f"555556{_RUN_ID[:4]}",
            "comment": "Gauntlet integration test — HVAC well-priced",
        },
    },
    {
        "name": "OOD booking (Remodeling — confidence < 0.5)",
        "pricing_payload": {
            "job_id":            f"staging-ood-{_RUN_ID}",
            "service_category":  "Remodeling",
            "zip_code":          "90210",
            "job_description":   "Full kitchen remodel, 400 sq ft, new cabinets and countertops",
            "deadline":          "Within 1 month",
            "original_estimate": 45000.0,
        },
        "booking_extras": {
            "name":    "Test Homeowner",
            "phone":   f"555557{_RUN_ID[:4]}",
            "comment": "Gauntlet integration test — OOD remodeling",
        },
    },
]

for case in STAGING_CASES:
    # Step 1: get our price estimate
    _, pricing = post(ENDPOINT, case["pricing_payload"], auth_headers())

    if not pricing.get("ok"):
        check(f"HouseAccount staging: {case['name']}", False,
              f"pricing step failed: {pricing}")
        continue

    lo  = int(round(pricing["estimate_lo"]))
    hi  = int(round(pricing["estimate_hi"]))
    mid = pricing["estimate_midpoint"]
    conf = pricing["confidence"]

    # Step 2: build HouseAccount booking payload
    # Required: name, zip, phone, summary
    # estimate: {min, max} in USD integers
    booking = {
        "name":    case["booking_extras"]["name"],
        "zip":     case["pricing_payload"]["zip_code"],
        "phone":   case["booking_extras"]["phone"],
        "summary": case["pricing_payload"]["job_description"][:200],
        "estimate": {"min": lo, "max": hi},
        "deadline": case["pricing_payload"].get("deadline", ""),
        "comment":  (
            f"{case['booking_extras']['comment']} | "
            f"midpoint=${mid:.0f} confidence={conf:.2f} "
            f"model={pricing.get('model_version','')}"
        ),
    }

    # Step 3: POST to HouseAccount staging
    status, body = ha_post("/api/bookings", booking)

    ok     = status in (200, 201)
    detail = f"HTTP {status} | our estimate ${lo}-${hi} (conf={conf:.2f})"
    if not ok:
        detail += f" | error: {body.get('error') or body}"
    check(f"HouseAccount staging: {case['name']}", ok, detail)


# ── Summary ────────────────────────────────────────────────────────────────

total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print(f"\n{'='*55}")
print(f"  {passed}/{total} passed  |  {failed} failed")
print(f"{'='*55}")

if failed:
    sys.exit(1)
