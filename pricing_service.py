"""
FastAPI pricing service.
Loads XGBoost models at startup, handles Haiku scope extraction + inference.
Records every estimate in SQLite; accepts final_price outcomes to close the
feedback loop and trigger background model retraining.

Usage:
    python3 -m uvicorn pricing_service:app --port 8001
"""
import asyncio
import csv
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import threading
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# Load .env if present (dev convenience — production uses real env vars)
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

import anthropic
import joblib
import numpy as np
import pgeocode
import xgboost as xgb
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

MODELS_DIR        = "models"
META_PATH         = os.path.join(MODELS_DIR, "meta.json")
DB_PATH           = os.path.join(MODELS_DIR, "outcomes.db")
FEATURES_CSV      = "features_enriched.csv"
# Trigger retrain when this many new labeled pairs have accumulated since
# the last retrain. Override via RETRAIN_THRESHOLD env var.
RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", "20"))

EXTRACTION_TOOL = {
    "name": "extract_scope",
    "description": "Extract scope features from a home service job description.",
    "input_schema": {
        "type": "object",
        "properties": {
            "labor_only":      {"type": "boolean",
                                "description": "True if customer supplies all parts/materials."},
            "task_count":      {"type": "integer",
                                "description": "Number of distinct tasks or items."},
            "complexity_tier": {"type": "string", "enum": ["low", "medium", "high"],
                                "description": "low=single simple task, medium=moderate, high=multi-trade."},
            "has_area_measure":{"type": "boolean",
                                "description": "True if sq ft, room count, or linear footage is mentioned."},
        },
        "required": ["labor_only", "task_count", "complexity_tier", "has_area_measure"],
    },
}
EXTRACTION_DEFAULTS = {
    "labor_only": False, "task_count": 1,
    "complexity_tier": "medium", "has_area_measure": False,
}


# ── App state ──────────────────────────────────────────────────────────────

class State:
    models:          dict     = {}
    meta:            dict     = {}
    client:          object   = None
    cache:           LRUCache = LRUCache(maxsize=2048)
    nomi:            object   = None
    db_lock:         object   = None   # threading.Lock for SQLite writes
    retrain_running: bool     = False

state = State()


# ── DB helpers ─────────────────────────────────────────────────────────────

def _init_db(path: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS estimates (
                job_id           TEXT    PRIMARY KEY,
                service_category TEXT    NOT NULL,
                zip_code         TEXT    NOT NULL,
                estimate_lo      REAL    NOT NULL,
                estimate_hi      REAL    NOT NULL,
                estimate_midpoint REAL   NOT NULL,
                confidence       REAL    NOT NULL,
                model_version    TEXT    NOT NULL,
                feature_vector   TEXT    NOT NULL,
                created_at       TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS outcomes (
                job_id       TEXT  PRIMARY KEY,
                final_price  REAL  NOT NULL,
                ape          REAL,
                recorded_at  TEXT  NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_outcomes_recorded
                ON outcomes(recorded_at);
        """)


def _store_estimate(job_id: str, category: str, zip_code: str,
                    lo: float, hi: float, mid: float, conf: float,
                    model_version: str, X: np.ndarray) -> None:
    try:
        with state.db_lock:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO estimates
                       (job_id, service_category, zip_code,
                        estimate_lo, estimate_hi, estimate_midpoint,
                        confidence, model_version, feature_vector, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (job_id, category, zip_code, lo, hi, mid, conf,
                     model_version, json.dumps(X[0].tolist()),
                     datetime.now(timezone.utc).isoformat()),
                )
    except Exception as e:
        log.error(f"Failed to store estimate {job_id}: {e}")


# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading model artifacts...")
    for q, tag in [(0.05, "q005"), (0.5, "q050"), (0.95, "q095")]:
        path = os.path.join(MODELS_DIR, f"xgb_{tag}.joblib")
        if not os.path.exists(path):
            raise RuntimeError(f"Model not found: {path}. Run train.py first.")
        state.models[q] = joblib.load(path)
    with open(META_PATH) as f:
        state.meta = json.load(f)
    log.info(f"Models loaded. Trained on {state.meta['n_train']} rows.")

    log.info("Loading pgeocode data...")
    state.nomi = pgeocode.Nominatim("us")
    log.info("pgeocode ready.")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    state.client = anthropic.AsyncAnthropic(api_key=api_key)
    log.info("Anthropic client ready.")

    state.db_lock = threading.Lock()
    _init_db(DB_PATH)
    log.info(f"Outcome DB ready at {DB_PATH}")

    yield
    log.info("Shutting down.")


app = FastAPI(title="HouseAccount Pricing Service", lifespan=lifespan)


# ── Error handlers ─────────────────────────────────────────────────────────

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


# ── Request / Response models ──────────────────────────────────────────────

class PricingRequest(BaseModel):
    job_id:               str
    service_category:     str
    zip_code:             str
    job_description:      str            = Field(max_length=4000)
    service_subtype:      Optional[str]  = None
    deadline:             Optional[str]  = None
    booking_month:        Optional[str]  = None
    original_estimate:    Optional[float] = Field(default=None, ge=1.0)
    original_estimate_lo: Optional[float] = None
    original_estimate_hi: Optional[float] = None
    job_status:           Optional[str]  = None

class PricingResponse(BaseModel):
    ok:                bool  = True
    job_id:            str
    estimate_lo:       float
    estimate_hi:       float
    estimate_midpoint: float
    confidence:        float = Field(ge=0.0, le=1.0)
    model_version:     str

class OutcomeRequest(BaseModel):
    job_id:      str
    final_price: float = Field(gt=0, description="Actual amount charged by the provider")

class OutcomeResponse(BaseModel):
    ok:          bool           = True
    job_id:      str
    ape:         Optional[float] = None  # absolute % error vs our midpoint; null if job_id unknown


# ── LLM scope extraction ───────────────────────────────────────────────────

async def extract_scope(description: str) -> dict:
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


# ── Feature building ───────────────────────────────────────────────────────

def _zip_income(zip_code: str) -> float:
    try:
        row        = state.nomi.query_postal_code(zip_code)
        state_name = row.get("state_name") if hasattr(row, "get") else row["state_name"]
        if str(state_name) != "nan":
            return _STATE_INCOME.get(str(state_name), _STATE_INCOME["US"]) / 100_000
    except Exception:
        pass
    return _STATE_INCOME["US"] / 100_000


def build_feature_vector(req: PricingRequest, scope: dict) -> np.ndarray:
    meta           = state.meta
    deadline_map   = meta["deadline_map"]
    complexity_map = meta["complexity_map"]
    categories     = meta["categories"]

    labor_only  = 1 if scope.get("labor_only")      else 0
    has_area    = 1 if scope.get("has_area_measure") else 0
    task_count  = min(int(scope.get("task_count", 1)), 20)
    complexity  = complexity_map.get(scope.get("complexity_tier", "medium"), 1)
    deadline    = deadline_map.get(req.deadline or "", 0)
    state_income = _zip_income(req.zip_code)

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
    well_priced  = set(state.meta.get("well_priced_categories", []))
    use_baseline = req.service_category in well_priced and req.original_estimate

    if use_baseline:
        lo  = req.original_estimate_lo  if req.original_estimate_lo  is not None else req.original_estimate * 0.8
        hi  = req.original_estimate_hi  if req.original_estimate_hi  is not None else req.original_estimate * 1.2
        mid = req.original_estimate
        return fix_intervals(lo, mid, hi)

    lo_raw  = float(state.models[0.05].predict(X)[0])
    mid_raw = float(state.models[0.5].predict(X)[0])
    hi_raw  = float(state.models[0.95].predict(X)[0])
    return fix_intervals(lo_raw, mid_raw, hi_raw)


# ── Retrain logic ──────────────────────────────────────────────────────────

def _xgb_params(quantile: float) -> dict:
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


def _retrain_sync() -> None:
    """
    Blocking retrain. Runs in a thread-pool executor so it never blocks
    the event loop. Combines original labeled CSV rows with all outcomes
    recorded in the DB, retrains three quantile XGBoost models, and
    atomically replaces the live model files.
    """
    try:
        meta           = state.meta
        deadline_map   = meta["deadline_map"]
        complexity_map = meta["complexity_map"]
        categories     = meta["categories"]

        # ── Load original labeled rows from CSV ────────────────────────────
        csv_rows = []
        if os.path.exists(FEATURES_CSV):
            with open(FEATURES_CSV) as f:
                for row in csv.DictReader(f):
                    if row.get("final_price"):
                        csv_rows.append(row)

        def _csv_features(row: dict) -> list:
            labor_only   = 1 if str(row.get("labor_only", "")).lower() == "true" else 0
            has_area     = 1 if str(row.get("has_area_measure", "")).lower() == "true" else 0
            task_count   = min(int(row.get("task_count") or 1), 20)
            complexity   = complexity_map.get(row.get("complexity_tier", "medium"), 1)
            deadline     = deadline_map.get(row.get("deadline", ""), 0)
            inc_raw      = row.get("state_median_income")
            state_income = float(inc_raw) / 100_000 if inc_raw else _STATE_INCOME["US"] / 100_000
            orig_est     = float(row["original_estimate"]) if row.get("original_estimate") else 0.0
            cat          = row.get("service_category", "")
            cat_vec      = [1 if cat == c else 0 for c in categories]
            return [labor_only, has_area, task_count, complexity,
                    deadline, state_income, orig_est, *cat_vec]

        X_csv = np.array([_csv_features(r) for r in csv_rows], dtype=np.float32)
        y_csv = np.array([float(r["final_price"]) for r in csv_rows], dtype=np.float32)

        # ── Load outcomes recorded since service started ───────────────────
        with sqlite3.connect(DB_PATH) as conn:
            db_rows = conn.execute("""
                SELECT e.feature_vector, o.final_price
                FROM outcomes o
                JOIN estimates e ON o.job_id = e.job_id
            """).fetchall()
            new_count = len(db_rows)

        if new_count == 0 and len(csv_rows) == 0:
            log.warning("Retrain skipped: no training data.")
            return

        X_parts = [X_csv] if len(X_csv) else []
        y_parts = [y_csv] if len(y_csv) else []
        if db_rows:
            X_db = np.array([json.loads(r[0]) for r in db_rows], dtype=np.float32)
            y_db = np.array([r[1] for r in db_rows], dtype=np.float32)
            X_parts.append(X_db)
            y_parts.append(y_db)

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        log.info(f"Retrain: {len(csv_rows)} CSV rows + {new_count} DB outcomes = {len(y)} total.")

        # ── Train ──────────────────────────────────────────────────────────
        new_models = {}
        for q in [0.05, 0.5, 0.95]:
            m = xgb.XGBRegressor(**_xgb_params(q))
            m.fit(X, y)
            new_models[q] = m

        # ── Atomic swap: write tmp files then os.rename ────────────────────
        tag_map = {0.05: "q005", 0.5: "q050", 0.95: "q095"}
        for q, m in new_models.items():
            tag      = tag_map[q]
            tmp_path = os.path.join(MODELS_DIR, f"_tmp_xgb_{tag}.joblib")
            dst_path = os.path.join(MODELS_DIR, f"xgb_{tag}.joblib")
            joblib.dump(m, tmp_path)
            os.rename(tmp_path, dst_path)

        # ── Hot-reload in service state (GIL protects dict assignment) ────
        state.models.update(new_models)
        state.meta["n_train"] = len(y)

        log.info(f"Retrain complete. New n_train={len(y)}. Models hot-swapped.")

    except Exception as e:
        log.error(f"Retrain failed: {e}", exc_info=True)
    finally:
        state.retrain_running = False


async def _maybe_retrain() -> None:
    """Fire-and-forget: check threshold and kick off retrain if needed."""
    if state.retrain_running:
        return
    try:
        with sqlite3.connect(DB_PATH) as conn:
            new_labeled = conn.execute(
                "SELECT COUNT(*) FROM outcomes o JOIN estimates e ON o.job_id = e.job_id"
            ).fetchone()[0]
        n_train_at_last_fit = state.meta.get("n_train", 0)
        if new_labeled >= RETRAIN_THRESHOLD and new_labeled > n_train_at_last_fit:
            log.info(f"Retrain threshold hit ({new_labeled} outcomes). Launching background retrain.")
            state.retrain_running = True
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, _retrain_sync)
    except Exception as e:
        log.error(f"Retrain check failed: {e}")


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post(
    "/.netlify/functions/pricing-estimate",
    response_model=PricingResponse,
    dependencies=[Depends(_require_bearer)],
)
async def predict(req: PricingRequest):
    scope       = await extract_scope(req.job_description)
    X           = build_feature_vector(req, scope)
    lo, mid, hi = route_predict(req, X)

    lo  = max(lo,  0.0)
    mid = max(mid, 1.0)
    hi  = max(hi,  mid)

    confidence    = compute_confidence(lo, hi, mid, req.service_category)
    model_version = "heila-v1.0.0"

    log.info(
        f"estimate job={req.job_id[:8]} cat={req.service_category} "
        f"lo={lo:.0f} mid={mid:.0f} hi={hi:.0f} conf={confidence:.2f} "
        f"labor_only={scope.get('labor_only')} tasks={scope.get('task_count')}"
    )

    _store_estimate(
        req.job_id, req.service_category, req.zip_code,
        round(lo, 2), round(hi, 2), round(mid, 2),
        confidence, model_version, X,
    )

    return PricingResponse(
        job_id=req.job_id,
        estimate_lo=round(lo, 2),
        estimate_hi=round(hi, 2),
        estimate_midpoint=round(mid, 2),
        confidence=confidence,
        model_version=model_version,
    )


@app.post(
    "/.netlify/functions/pricing-outcome",
    response_model=OutcomeResponse,
    dependencies=[Depends(_require_bearer)],
)
async def record_outcome(req: OutcomeRequest):
    """
    Called by HouseAccount when a job closes with a known final_price.
    Stores the label, computes APE against our stored midpoint, and
    triggers a background model retrain when the threshold is reached.
    """
    ape = None
    try:
        with state.db_lock:
            with sqlite3.connect(DB_PATH) as conn:
                row = conn.execute(
                    "SELECT estimate_midpoint FROM estimates WHERE job_id = ?",
                    (req.job_id,),
                ).fetchone()

                if row:
                    mid = row[0]
                    if mid and mid > 0:
                        ape = round(abs(req.final_price - mid) / req.final_price * 100, 2)

                conn.execute(
                    """INSERT OR REPLACE INTO outcomes
                       (job_id, final_price, ape, recorded_at)
                       VALUES (?, ?, ?, ?)""",
                    (req.job_id, req.final_price, ape,
                     datetime.now(timezone.utc).isoformat()),
                )
    except Exception as e:
        log.error(f"Failed to record outcome {req.job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to store outcome")

    log.info(
        f"outcome job={req.job_id[:8]} "
        f"final=${req.final_price:.0f} ape={ape}%"
    )

    asyncio.create_task(_maybe_retrain())

    return OutcomeResponse(job_id=req.job_id, ape=ape)


@app.post("/retrain", dependencies=[Depends(_require_bearer)])
async def trigger_retrain():
    """Manual retrain trigger for ops use. Runs in background."""
    if state.retrain_running:
        return {"ok": True, "message": "Retrain already in progress"}
    state.retrain_running = True
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _retrain_sync)
    return {"ok": True, "message": "Retrain started"}


@app.get("/metrics")
def metrics():
    """
    Rolling accuracy stats from the outcome DB.
    Useful for HouseAccount to monitor model health in production.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            total_estimates = conn.execute(
                "SELECT COUNT(*) FROM estimates"
            ).fetchone()[0]
            total_outcomes = conn.execute(
                "SELECT COUNT(*) FROM outcomes"
            ).fetchone()[0]
            # Rolling MAPE over last 90 days
            mape_row = conn.execute("""
                SELECT AVG(ape), COUNT(*)
                FROM outcomes
                WHERE ape IS NOT NULL
                  AND recorded_at >= datetime('now', '-90 days')
            """).fetchone()
            rolling_mape  = round(mape_row[0], 2) if mape_row[0] is not None else None
            outcomes_90d  = mape_row[1]
            # Per-category rolling MAPE
            per_cat = conn.execute("""
                SELECT e.service_category, AVG(o.ape), COUNT(*)
                FROM outcomes o
                JOIN estimates e ON o.job_id = e.job_id
                WHERE o.ape IS NOT NULL
                  AND o.recorded_at >= datetime('now', '-90 days')
                GROUP BY e.service_category
                ORDER BY AVG(o.ape)
            """).fetchall()
        return {
            "ok":              True,
            "total_estimates": total_estimates,
            "total_outcomes":  total_outcomes,
            "rolling_mape_90d": rolling_mape,
            "outcomes_90d":    outcomes_90d,
            "n_train":         state.meta.get("n_train"),
            "retrain_running": state.retrain_running,
            "retrain_threshold": RETRAIN_THRESHOLD,
            "per_category_mape_90d": [
                {"category": r[0], "mape": round(r[1], 1), "n": r[2]}
                for r in per_cat
            ],
        }
    except Exception as e:
        log.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")


@app.get("/health")
def health():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            n_outcomes = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
    except Exception:
        n_outcomes = -1
    return {
        "ok":             True,
        "models_loaded":  len(state.models),
        "cache_size":     len(state.cache),
        "n_outcomes":     n_outcomes,
        "retrain_running": state.retrain_running,
    }
