"""
FastAPI pricing service — step 4.
Loads XGBoost models at startup, handles Haiku scope extraction + inference.

Usage:
    ANTHROPIC_API_KEY=your_key python3 -m uvicorn pricing_service:app --port 8000
"""
import asyncio
import hashlib
import hmac
import json
import logging
import os
import warnings
from pathlib import Path

# Load .env if present (dev convenience — production uses real env vars)
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())
warnings.filterwarnings("ignore")
from contextlib import asynccontextmanager
from typing import Optional

import anthropic
import joblib
import numpy as np
import pgeocode
from cachetools import LRUCache
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 2023 ACS 1-year estimates — same source as enrich_zip.py
_STATE_INCOME = {
    "Maryland": 101_710, "New Jersey": 101_050, "Massachusetts": 98_700,
    "Hawaii": 97_600, "Connecticut": 91_400, "California": 91_100,
    "Washington": 90_300, "Colorado": 87_900, "Virginia": 87_600,
    "Minnesota": 85_700, "New Hampshire": 84_900, "Utah": 84_800,
    "Alaska": 83_000, "New York": 80_900, "Illinois": 78_000,
    "Oregon": 77_100, "Rhode Island": 76_400, "Delaware": 76_200,
    "Wisconsin": 74_500, "Georgia": 73_600, "Arizona": 72_600,
    "Nevada": 72_400, "Texas": 72_300, "North Carolina": 71_500,
    "Florida": 70_900, "Pennsylvania": 70_600, "Michigan": 70_300,
    "Idaho": 70_100, "Ohio": 68_700, "Indiana": 68_400,
    "Tennessee": 66_600, "Montana": 66_000, "South Carolina": 65_300,
    "Vermont": 65_100, "Iowa": 65_100, "Nebraska": 65_000,
    "Missouri": 64_800, "Kansas": 64_600, "Wyoming": 64_200,
    "North Dakota": 63_400, "South Dakota": 62_900, "Maine": 62_900,
    "Oklahoma": 62_200, "Alabama": 59_600, "Kentucky": 59_300,
    "Louisiana": 58_300, "New Mexico": 58_200, "Arkansas": 57_600,
    "West Virginia": 55_900, "Mississippi": 53_600,
    "District of Columbia": 101_700, "Puerto Rico": 24_000,
    "US": 75_149,
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────

MODELS_DIR = "models"
META_PATH  = os.path.join(MODELS_DIR, "meta.json")

EXTRACTION_TOOL = {
    "name": "extract_scope",
    "description": "Extract scope features from a home service job description.",
    "input_schema": {
        "type": "object",
        "properties": {
            "labor_only":     {"type": "boolean",
                               "description": "True if customer supplies all parts/materials. Examples: 'we supply', 'you supply', assembly of items the customer owns."},
            "task_count":     {"type": "integer",
                               "description": "Number of distinct tasks or items. Single repair=1, 'assemble shelf + mount lights'=2."},
            "complexity_tier":{"type": "string", "enum": ["low","medium","high"],
                               "description": "low=single simple task, medium=multiple tasks or moderate skill, high=multi-trade or large scope."},
            "has_area_measure":{"type": "boolean",
                                "description": "True if square footage, room count, or linear footage is mentioned."},
        },
        "required": ["labor_only","task_count","complexity_tier","has_area_measure"],
    },
}
EXTRACTION_DEFAULTS = {
    "labor_only": False, "task_count": 1,
    "complexity_tier": "medium", "has_area_measure": False,
}
# ── App state ──────────────────────────────────────────────────────────────

class State:
    models: dict        = {}
    meta:   dict        = {}
    client: object      = None
    cache:  LRUCache    = LRUCache(maxsize=2048)
    nomi:   object      = None

state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models
    log.info("Loading model artifacts...")
    for q, tag in [(0.05,"q005"), (0.5,"q050"), (0.95,"q095")]:
        path = os.path.join(MODELS_DIR, f"xgb_{tag}.joblib")
        if not os.path.exists(path):
            raise RuntimeError(f"Model not found: {path}. Run train.py first.")
        state.models[q] = joblib.load(path)
    with open(META_PATH) as f:
        state.meta = json.load(f)
    log.info(f"Models loaded. Trained on {state.meta['n_train']} rows.")

    # pgeocode for ZIP→state income lookup (Bug 2 fix)
    log.info("Loading pgeocode data...")
    state.nomi = pgeocode.Nominatim("us")
    log.info("pgeocode ready.")

    # Anthropic client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    state.client = anthropic.AsyncAnthropic(api_key=api_key)
    log.info("Anthropic client ready.")
    yield
    log.info("Shutting down.")


app = FastAPI(title="HouseAccount Pricing Service", lifespan=lifespan)


# ── Error handlers (PRD-mandated shapes) ──────────────────────────────────

@app.exception_handler(RequestValidationError)
async def _validation_error(request, exc):
    for err in exc.errors():
        if err.get("type") == "json_invalid":
            return JSONResponse(status_code=400, content={"error": "Malformed JSON"})
    for err in exc.errors():
        if err.get("type") == "missing":
            field = err["loc"][-1] if err.get("loc") else "field"
            return JSONResponse(status_code=400, content={"error": f"{field} required"})
    return JSONResponse(status_code=400, content={"error": "Bad request"})

@app.exception_handler(HTTPException)
async def _http_error(request, exc):
    detail = "Method not allowed" if exc.status_code == 405 else exc.detail
    return JSONResponse(status_code=exc.status_code, content={"error": detail})

@app.exception_handler(Exception)
async def _server_error(request, exc):
    log.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ── Auth ───────────────────────────────────────────────────────────────────

def _require_bearer(authorization: str = Header(default="")):
    expected = os.environ.get("GAUNTLET_PRICING_SECRET", "")
    if not expected or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization[len("Bearer "):].encode()
    if not hmac.compare_digest(token, expected.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Request / Response ─────────────────────────────────────────────────────

class PricingRequest(BaseModel):
    job_id:             str
    service_category:   str
    zip_code:           str
    job_description:    str = Field(max_length=4000)
    service_subtype:    Optional[str]   = None
    deadline:           Optional[str]   = None
    booking_month:      Optional[str]   = None
    original_estimate:  Optional[float] = Field(default=None, ge=1.0)
    original_estimate_lo: Optional[float] = None
    original_estimate_hi: Optional[float] = None
    job_status:         Optional[str]   = None

class PricingResponse(BaseModel):
    ok:               bool  = True
    job_id:           str
    estimate_lo:      float
    estimate_hi:      float
    estimate_midpoint:float
    confidence:       float = Field(ge=0.0, le=1.0)
    model_version:    str


# ── Haiku extraction ───────────────────────────────────────────────────────

async def extract_scope(description: str) -> dict:
    """Extract scope features. Cached by description hash."""
    key = hashlib.sha256(description.encode()).hexdigest()
    if key in state.cache:
        return state.cache[key]

    for attempt in range(2):
        try:
            resp = await state.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                tools=[EXTRACTION_TOOL],
                tool_choice={"type": "tool", "name": "extract_scope"},
                messages=[{"role": "user",
                           "content": f"Extract scope features:\n\n{description}"}],
                timeout=5.0,
            )
            for block in resp.content:
                if block.type == "tool_use":
                    state.cache[key] = block.input
                    return block.input
        except Exception as e:
            if attempt == 0:
                log.warning(f"Haiku extraction attempt 1 failed: {e}, retrying...")
                await asyncio.sleep(1.0)
            else:
                log.error(f"Haiku extraction failed after 2 attempts: {e}")

    log.warning("Using extraction defaults.")
    return EXTRACTION_DEFAULTS


# ── Feature builder ────────────────────────────────────────────────────────

def _zip_income(zip_code: str) -> float:
    """ZIP → normalised state median income. Falls back to national median."""
    try:
        row        = state.nomi.query_postal_code(zip_code)
        state_name = row.get("state_name") if hasattr(row, "get") else row["state_name"]
        if str(state_name) != "nan":
            return _STATE_INCOME.get(str(state_name), _STATE_INCOME["US"]) / 100_000
    except Exception:
        pass
    return _STATE_INCOME["US"] / 100_000


def build_feature_vector(req: PricingRequest, scope: dict) -> np.ndarray:
    meta = state.meta
    deadline_map  = meta["deadline_map"]
    complexity_map = meta["complexity_map"]
    categories     = meta["categories"]

    labor_only  = 1 if scope.get("labor_only")      else 0
    has_area    = 1 if scope.get("has_area_measure") else 0
    task_count  = min(int(scope.get("task_count", 1)), 20)
    complexity  = complexity_map.get(scope.get("complexity_tier","medium"), 1)
    deadline    = deadline_map.get(req.deadline or "", 0)

    # Bug 2 fix: look up actual state income for the ZIP instead of using national median
    state_income = _zip_income(req.zip_code)

    # Bug 1 fix: use per-category training median when original_estimate is absent
    # (feeding 0 breaks the model's top feature; category median is in-distribution)
    if req.original_estimate:
        orig_est = req.original_estimate
    else:
        medians  = meta.get("category_estimate_medians", {})
        orig_est = medians.get(req.service_category, medians.get("overall", 300.0))

    cat_vec = [1 if req.service_category == c else 0 for c in categories]
    return np.array([[
        labor_only, has_area, task_count, complexity,
        deadline, state_income, orig_est,
        *cat_vec,
    ]], dtype=np.float32)


# ── Inference ──────────────────────────────────────────────────────────────

def fix_intervals(lo, mid, hi):
    mid = float(mid)
    lo  = float(min(lo, mid))
    hi  = float(max(hi, mid))
    return lo, mid, hi


def compute_confidence(lo: float, hi: float, mid: float, category: str) -> float:
    meta = state.meta
    if mid <= 0:
        return 0.3
    interval_ratio = (hi - lo) / mid
    base = 1.0 / (1.0 + interval_ratio)

    cap = 1.0
    if mid > 5000:
        cap = min(cap, 0.40)
    median_interval = meta.get("training_median_interval", 212)
    if (hi - lo) > 3 * median_interval:
        cap = min(cap, 0.45)
    if category not in set(meta.get("production_categories", [])):
        cap = min(cap, 0.40)

    return round(min(base, cap), 4)


def route_predict(req: PricingRequest, X: np.ndarray) -> tuple[float, float, float]:
    """Route to baseline or model based on category."""
    well_priced = set(state.meta.get("well_priced_categories", []))
    use_baseline = req.service_category in well_priced and req.original_estimate

    if use_baseline:
        # Trust original_estimate directly for well-priced categories
        lo  = req.original_estimate_lo  if req.original_estimate_lo  is not None else req.original_estimate * 0.8
        hi  = req.original_estimate_hi  if req.original_estimate_hi  is not None else req.original_estimate * 1.2
        mid = req.original_estimate
        return fix_intervals(lo, mid, hi)

    # Hard categories (Handyman, Plumbing, etc.) → XGBoost
    lo_raw  = float(state.models[0.05].predict(X)[0])
    mid_raw = float(state.models[0.5].predict(X)[0])
    hi_raw  = float(state.models[0.95].predict(X)[0])
    return fix_intervals(lo_raw, mid_raw, hi_raw)


# ── Endpoint ───────────────────────────────────────────────────────────────

@app.post("/.netlify/functions/pricing-estimate", response_model=PricingResponse, dependencies=[Depends(_require_bearer)])
async def predict(req: PricingRequest):
    scope = await extract_scope(req.job_description)
    X     = build_feature_vector(req, scope)
    lo, mid, hi = route_predict(req, X)

    # Clamp to non-negative
    lo  = max(lo,  0.0)
    mid = max(mid, 1.0)
    hi  = max(hi,  mid)

    confidence = compute_confidence(lo, hi, mid, req.service_category)

    log.info(
        f"job={req.job_id[:8]} cat={req.service_category} "
        f"lo={lo:.0f} mid={mid:.0f} hi={hi:.0f} conf={confidence:.2f} "
        f"labor_only={scope.get('labor_only')} tasks={scope.get('task_count')}"
    )

    return PricingResponse(
        job_id=req.job_id,
        estimate_lo=round(lo, 2),
        estimate_hi=round(hi, 2),
        estimate_midpoint=round(mid, 2),
        confidence=confidence,
        model_version="heila-v1.0.0",
    )


@app.get("/health")
def health():
    return {"ok": True, "models_loaded": len(state.models), "cache_size": len(state.cache)}
