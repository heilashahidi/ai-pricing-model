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
import pickle
import sqlite3
import threading
import time
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

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
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 2023 ACS 1-year estimates
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

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _log(event: str, **fields) -> None:
    log.info(json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                         "event": event, **fields}))


# ── Constants ──────────────────────────────────────────────────────────────

MODELS_DIR        = "models"
META_PATH         = os.path.join(MODELS_DIR, "meta.json")
DB_PATH           = os.path.join(MODELS_DIR, "outcomes.db")
CACHE_PATH        = os.path.join(MODELS_DIR, "extraction_cache.pkl")
FEATURES_CSV      = "features_enriched.csv"
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


# ── Reliability: circuit breaker + rate limiter + API key store ────────────

class _CircuitBreaker:
    """Open after N consecutive LLM failures; half-open after timeout."""
    def __init__(self, threshold: int = 5, timeout: float = 60.0):
        self._threshold = threshold
        self._timeout   = timeout
        self._failures  = 0
        self._opened_at: Optional[float] = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self._timeout:
            self._opened_at = None  # half-open: try once
            return False
        return True

    def on_success(self) -> None:
        self._failures  = 0
        self._opened_at = None

    def on_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold and self._opened_at is None:
            self._opened_at = time.monotonic()
            _log("circuit_open", service="anthropic",
                 failures=self._failures, timeout_s=self._timeout)


class _RateLimiter:
    """Token-bucket per key name. Thread-safe. Rate = requests/minute."""
    def __init__(self):
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def check(self, name: str, rate_per_min: int) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(name, (float(rate_per_min), now))
            tokens = min(float(rate_per_min),
                         tokens + (now - last) * rate_per_min / 60.0)
            if tokens < 1.0:
                self._buckets[name] = (tokens, now)
                return False
            self._buckets[name] = (tokens - 1.0, now)
            return True


class _ApiKeyStore:
    """
    Multi-tenant API key registry. Supports two env formats:

    Single key (backward compat):
        GAUNTLET_PRICING_SECRET=<hex>

    Multi-tenant:
        API_KEYS=[{"key":"<hex>","name":"prod","rate_limit":600}, ...]

    If both are set, API_KEYS takes precedence.
    """
    def __init__(self):
        self._keys: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        raw = os.environ.get("API_KEYS", "")
        if raw:
            try:
                for e in json.loads(raw):
                    self._keys[e["key"]] = {
                        "name":       e.get("name", "unknown"),
                        "rate_limit": int(e.get("rate_limit", 60)),
                    }
                _log("keys_loaded", count=len(self._keys), source="API_KEYS")
                return
            except Exception as ex:
                _log("keys_load_error", error=str(ex))

        secret = os.environ.get("GAUNTLET_PRICING_SECRET", "")
        if secret:
            self._keys[secret] = {"name": "default", "rate_limit": 60}
            _log("keys_loaded", count=1, source="GAUNTLET_PRICING_SECRET")

    def authenticate(self, token: str) -> Optional[dict]:
        for key, meta in self._keys.items():
            if hmac.compare_digest(token.encode(), key.encode()):
                return meta
        return None

    def reload(self) -> int:
        self._keys.clear()
        self._load()
        return len(self._keys)


# ── ZIP income cache ───────────────────────────────────────────────────────

_zip_income_cache: dict[str, float] = {}


def _compute_zip_income(zip_code: str, nomi) -> float:
    try:
        row        = nomi.query_postal_code(zip_code)
        state_name = row.get("state_name") if hasattr(row, "get") else row["state_name"]
        if str(state_name) != "nan":
            return _STATE_INCOME.get(str(state_name), _STATE_INCOME["US"]) / 100_000
    except Exception:
        pass
    return _STATE_INCOME["US"] / 100_000


def _zip_income(zip_code: str) -> float:
    if zip_code not in _zip_income_cache:
        _zip_income_cache[zip_code] = _compute_zip_income(zip_code, state.nomi)
    return _zip_income_cache[zip_code]


# ── App state ──────────────────────────────────────────────────────────────

class State:
    models:          dict            = {}
    meta:            dict            = {}
    client:          object          = None
    cache:           LRUCache        = LRUCache(maxsize=2048)
    nomi:            object          = None
    db_lock:         object          = None
    retrain_running: bool            = False
    circuit_breaker: object          = None  # _CircuitBreaker, set in lifespan
    rate_limiter:    object          = None  # _RateLimiter, set in lifespan
    key_store:       object          = None  # _ApiKeyStore, set in lifespan

state = State()


# ── DB helpers ─────────────────────────────────────────────────────────────

def _init_db(path: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS estimates (
                job_id            TEXT PRIMARY KEY,
                service_category  TEXT NOT NULL,
                zip_code          TEXT NOT NULL,
                estimate_lo       REAL NOT NULL,
                estimate_hi       REAL NOT NULL,
                estimate_midpoint REAL NOT NULL,
                confidence        REAL NOT NULL,
                model_version     TEXT NOT NULL,
                feature_vector    TEXT NOT NULL,
                created_at        TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS outcomes (
                job_id       TEXT PRIMARY KEY,
                final_price  REAL NOT NULL,
                ape          REAL,
                recorded_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_outcomes_recorded
                ON outcomes(recorded_at);
        """)


def _lookup_stored_estimate(job_id: str) -> Optional["PricingResponse"]:
    """Return the stored estimate for job_id if it exists (idempotency)."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                """SELECT estimate_lo, estimate_hi, estimate_midpoint,
                          confidence, model_version
                   FROM estimates WHERE job_id = ?""",
                (job_id,),
            ).fetchone()
            if row:
                return PricingResponse(
                    job_id=job_id,
                    estimate_lo=row[0], estimate_hi=row[1],
                    estimate_midpoint=row[2], confidence=row[3],
                    model_version=row[4],
                )
    except Exception:
        pass
    return None


def _store_estimate(job_id: str, category: str, zip_code: str,
                    lo: float, hi: float, mid: float, conf: float,
                    model_version: str, X: np.ndarray) -> None:
    if state.db_lock is None:
        return
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
        _log("db_error", op="store_estimate", job_id=job_id, error=str(e))


# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models
    _log("startup", phase="models")
    for q, tag in [(0.05, "q005"), (0.5, "q050"), (0.95, "q095")]:
        path = os.path.join(MODELS_DIR, f"xgb_{tag}.joblib")
        if not os.path.exists(path):
            raise RuntimeError(f"Model not found: {path}. Run train.py first.")
        state.models[q] = joblib.load(path)
    with open(META_PATH) as f:
        state.meta = json.load(f)
    _log("models_loaded", n_train=state.meta["n_train"],
         version=state.meta.get("model_version", "heila-v1.0.0"))

    # Extraction cache: restore from disk
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                state.cache.update(pickle.load(f))
            _log("cache_restored", size=len(state.cache))
        except Exception as e:
            _log("cache_restore_failed", error=str(e))

    # pgeocode
    _log("startup", phase="pgeocode")
    state.nomi = pgeocode.Nominatim("us")

    # Pre-warm ZIP income cache from training data
    if os.path.exists(FEATURES_CSV):
        with open(FEATURES_CSV) as f:
            for row in csv.DictReader(f):
                z = row.get("zip_code", "")
                if z and z not in _zip_income_cache:
                    _zip_income_cache[z] = _compute_zip_income(z, state.nomi)
    _log("zip_cache_warmed", size=len(_zip_income_cache))

    # Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    state.client = anthropic.AsyncAnthropic(api_key=api_key)

    # Reliability primitives
    state.circuit_breaker = _CircuitBreaker(threshold=5, timeout=60.0)
    state.rate_limiter    = _RateLimiter()
    state.key_store       = _ApiKeyStore()

    # DB
    state.db_lock = threading.Lock()
    _init_db(DB_PATH)
    _log("startup", phase="ready")

    yield

    # Persist extraction cache to disk on clean shutdown
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(dict(state.cache), f)
        _log("cache_persisted", size=len(state.cache))
    except Exception as e:
        _log("cache_persist_failed", error=str(e))

    _log("shutdown")


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
    if exc.status_code == 405:
        return JSONResponse(status_code=405, content={"error": "Method not allowed"})
    if exc.status_code == 429:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "retry_after": 60},
            headers={"Retry-After": "60"},
        )
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.exception_handler(Exception)
async def _server_error(request, exc):
    _log("unhandled_error", error=str(exc))
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ── Auth ───────────────────────────────────────────────────────────────────

def _get_key_meta(authorization: str = Header(default="")) -> dict:
    """
    Validates Bearer token against the key store.
    Returns key metadata (name, rate_limit) on success.
    Raises 401 on bad/missing token, 429 on rate limit exceeded.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization[len("Bearer "):]

    ks = state.key_store
    if ks is None:
        # Fallback during tests / before lifespan
        expected = os.environ.get("GAUNTLET_PRICING_SECRET", "")
        if not expected or not hmac.compare_digest(token.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="Unauthorized")
        return {"name": "default", "rate_limit": 60}

    meta = ks.authenticate(token)
    if meta is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    rl = state.rate_limiter
    if rl and not rl.check(meta["name"], meta["rate_limit"]):
        _log("rate_limited", key_name=meta["name"])
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    return meta


# ── Pydantic models ────────────────────────────────────────────────────────

class PricingRequest(BaseModel):
    job_id:               str
    service_category:     str
    zip_code:             str
    job_description:      str             = Field(max_length=4000)
    service_subtype:      Optional[str]   = None
    deadline:             Optional[str]   = None
    booking_month:        Optional[str]   = None
    original_estimate:    Optional[float] = Field(default=None, ge=1.0)
    original_estimate_lo: Optional[float] = None
    original_estimate_hi: Optional[float] = None
    job_status:           Optional[str]   = None

class PricingResponse(BaseModel):
    ok:                bool  = True
    job_id:            str
    estimate_lo:       float
    estimate_hi:       float
    estimate_midpoint: float
    confidence:        float = Field(ge=0.0, le=1.0)
    model_version:     str

class BatchPricingRequest(BaseModel):
    estimates: list[PricingRequest] = Field(min_length=1, max_length=50)

class BatchItem(BaseModel):
    ok:                bool            = True
    job_id:            Optional[str]   = None
    estimate_lo:       Optional[float] = None
    estimate_hi:       Optional[float] = None
    estimate_midpoint: Optional[float] = None
    confidence:        Optional[float] = None
    model_version:     Optional[str]   = None
    error:             Optional[str]   = None

class BatchPricingResponse(BaseModel):
    ok:      bool            = True
    results: list[BatchItem]

class OutcomeRequest(BaseModel):
    job_id:      str
    final_price: float = Field(gt=0)

class OutcomeResponse(BaseModel):
    ok:          bool           = True
    job_id:      str
    ape:         Optional[float] = None


# ── LLM scope extraction ───────────────────────────────────────────────────

async def extract_scope(description: str) -> dict:
    """
    Extract scope features via Claude Haiku. Returns cached result immediately.
    Falls back to EXTRACTION_DEFAULTS on circuit-open or double failure.
    Failures do NOT populate the cache.
    """
    key = hashlib.sha256(description.encode()).hexdigest()
    if key in state.cache:
        return state.cache[key]

    cb = state.circuit_breaker
    if cb and cb.is_open:
        return EXTRACTION_DEFAULTS

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
                    if cb:
                        cb.on_success()
                    state.cache[key] = block.input
                    return block.input
        except Exception as e:
            if cb:
                cb.on_failure()
            if attempt == 0:
                _log("llm_retry", error=str(e))
                await asyncio.sleep(1.0)
            else:
                _log("llm_failed", error=str(e))

    return EXTRACTION_DEFAULTS


# ── Feature building ───────────────────────────────────────────────────────

def build_feature_vector(req: PricingRequest, scope: dict) -> np.ndarray:
    meta           = state.meta
    deadline_map   = meta["deadline_map"]
    complexity_map = meta["complexity_map"]
    categories     = meta["categories"]

    labor_only   = 1 if scope.get("labor_only")      else 0
    has_area     = 1 if scope.get("has_area_measure") else 0
    task_count   = min(int(scope.get("task_count", 1)), 20)
    complexity   = complexity_map.get(scope.get("complexity_tier", "medium"), 1)
    deadline     = deadline_map.get(req.deadline or "", 0)
    state_income = _zip_income(req.zip_code)

    if req.original_estimate:
        orig_est = req.original_estimate
    else:
        medians  = meta.get("category_estimate_medians", {})
        orig_est = medians.get(req.service_category, medians.get("overall", 300.0))

    cat_vec = [1 if req.service_category == c else 0 for c in categories]
    return np.array([[
        labor_only, has_area, task_count, complexity,
        deadline, state_income, orig_est, *cat_vec,
    ]], dtype=np.float32)


# ── Inference ──────────────────────────────────────────────────────────────

def fix_intervals(lo, mid, hi):
    mid = float(mid)
    return float(min(lo, mid)), mid, float(max(hi, mid))


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
        return fix_intervals(lo, req.original_estimate, hi)

    lo_raw  = float(state.models[0.05].predict(X)[0])
    mid_raw = float(state.models[0.5].predict(X)[0])
    hi_raw  = float(state.models[0.95].predict(X)[0])
    return fix_intervals(lo_raw, mid_raw, hi_raw)


# ── Retrain ────────────────────────────────────────────────────────────────

def _bump_version(version: str) -> str:
    try:
        prefix, semver = version.rsplit("-v", 1)
        major, minor, patch = semver.split(".")
        return f"{prefix}-v{major}.{minor}.{int(patch)+1}"
    except Exception:
        return f"{version}.1"


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
    Blocking retrain — runs in thread-pool executor.
    Merges original labeled CSV rows + DB outcome rows, retrains three
    quantile models, atomically swaps model files, hot-reloads state.
    """
    try:
        meta           = state.meta
        deadline_map   = meta["deadline_map"]
        complexity_map = meta["complexity_map"]
        categories     = meta["categories"]

        # Original labeled CSV rows
        csv_rows = []
        if os.path.exists(FEATURES_CSV):
            with open(FEATURES_CSV) as f:
                csv_rows = [r for r in csv.DictReader(f) if r.get("final_price")]

        def _csv_feat(row: dict) -> list:
            labor_only   = 1 if str(row.get("labor_only", "")).lower() == "true" else 0
            has_area     = 1 if str(row.get("has_area_measure", "")).lower() == "true" else 0
            task_count   = min(int(row.get("task_count") or 1), 20)
            complexity   = complexity_map.get(row.get("complexity_tier", "medium"), 1)
            deadline     = deadline_map.get(row.get("deadline", ""), 0)
            inc          = row.get("state_median_income")
            state_income = float(inc) / 100_000 if inc else _STATE_INCOME["US"] / 100_000
            orig_est     = float(row["original_estimate"]) if row.get("original_estimate") else 0.0
            cat          = row.get("service_category", "")
            cat_vec      = [1 if cat == c else 0 for c in categories]
            return [labor_only, has_area, task_count, complexity,
                    deadline, state_income, orig_est, *cat_vec]

        X_parts, y_parts = [], []
        if csv_rows:
            X_parts.append(np.array([_csv_feat(r) for r in csv_rows], dtype=np.float32))
            y_parts.append(np.array([float(r["final_price"]) for r in csv_rows], dtype=np.float32))

        # New outcome rows (feature vectors already stored)
        with sqlite3.connect(DB_PATH) as conn:
            db_rows = conn.execute(
                "SELECT e.feature_vector, o.final_price FROM outcomes o "
                "JOIN estimates e ON o.job_id = e.job_id"
            ).fetchall()
        if db_rows:
            X_parts.append(np.array([json.loads(r[0]) for r in db_rows], dtype=np.float32))
            y_parts.append(np.array([r[1] for r in db_rows], dtype=np.float32))

        if not X_parts:
            _log("retrain_skipped", reason="no_data")
            return

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        _log("retrain_start", csv_rows=len(csv_rows),
             db_rows=len(db_rows), total=len(y))

        # Train
        new_models = {}
        for q in [0.05, 0.5, 0.95]:
            m = xgb.XGBRegressor(**_xgb_params(q))
            m.fit(X, y)
            new_models[q] = m

        # Atomic swap: write tmp → rename
        tag_map = {0.05: "q005", 0.5: "q050", 0.95: "q095"}
        for q, m in new_models.items():
            tag = tag_map[q]
            tmp = os.path.join(MODELS_DIR, f"_tmp_xgb_{tag}.joblib")
            dst = os.path.join(MODELS_DIR, f"xgb_{tag}.joblib")
            joblib.dump(m, tmp)
            os.rename(tmp, dst)

        # Hot-reload state (GIL protects dict assignment)
        state.models.update(new_models)
        new_version = _bump_version(state.meta.get("model_version", "heila-v1.0.0"))
        state.meta["n_train"]      = len(y)
        state.meta["model_version"] = new_version
        with open(META_PATH, "w") as f:
            json.dump(state.meta, f, indent=2)

        _log("retrain_complete", n_train=len(y), new_version=new_version)

    except Exception as e:
        _log("retrain_error", error=str(e))
    finally:
        state.retrain_running = False


async def _maybe_retrain() -> None:
    if state.retrain_running:
        return
    try:
        with sqlite3.connect(DB_PATH) as conn:
            new_labeled = conn.execute(
                "SELECT COUNT(*) FROM outcomes o JOIN estimates e ON o.job_id = e.job_id"
            ).fetchone()[0]
        n_at_last_fit = state.meta.get("n_train", 0)
        if new_labeled >= RETRAIN_THRESHOLD and new_labeled > n_at_last_fit:
            _log("retrain_triggered", new_labeled=new_labeled)
            state.retrain_running = True
            asyncio.get_event_loop().run_in_executor(None, _retrain_sync)
    except Exception as e:
        _log("retrain_check_error", error=str(e))


# ── Core inference (shared between single + batch) ─────────────────────────

async def _predict_one(req: PricingRequest, key_name: str) -> PricingResponse:
    t0 = time.monotonic()

    # Idempotency: same job_id → return stored result
    stored = _lookup_stored_estimate(req.job_id)
    if stored:
        _log("estimate_idempotent", job_id=req.job_id, key_name=key_name,
             latency_ms=round((time.monotonic() - t0) * 1000))
        return stored

    desc_hash   = hashlib.sha256(req.job_description.encode()).hexdigest()
    scope_cached = desc_hash in state.cache

    scope       = await extract_scope(req.job_description)
    X           = build_feature_vector(req, scope)
    lo, mid, hi = route_predict(req, X)

    lo  = max(lo,  0.0)
    mid = max(mid, 1.0)
    hi  = max(hi,  mid)

    confidence    = compute_confidence(lo, hi, mid, req.service_category)
    model_version = state.meta.get("model_version", "heila-v1.0.0")
    latency_ms    = round((time.monotonic() - t0) * 1000)

    _log("estimate",
         job_id=req.job_id[:8], category=req.service_category, zip=req.zip_code,
         lo=round(lo), mid=round(mid), hi=round(hi), confidence=confidence,
         latency_ms=latency_ms, model_version=model_version,
         scope_cached=scope_cached, key_name=key_name,
         labor_only=scope.get("labor_only"), tasks=scope.get("task_count"))

    _store_estimate(req.job_id, req.service_category, req.zip_code,
                    round(lo, 2), round(hi, 2), round(mid, 2),
                    confidence, model_version, X)

    return PricingResponse(
        job_id=req.job_id,
        estimate_lo=round(lo, 2),
        estimate_hi=round(hi, 2),
        estimate_midpoint=round(mid, 2),
        confidence=confidence,
        model_version=model_version,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/.netlify/functions/pricing-estimate", response_model=PricingResponse)
async def predict(req: PricingRequest,
                  key_meta: dict = Depends(_get_key_meta)):
    return await _predict_one(req, key_meta["name"])


@app.post("/.netlify/functions/pricing-estimate-batch",
          response_model=BatchPricingResponse)
async def predict_batch(req: BatchPricingRequest,
                        key_meta: dict = Depends(_get_key_meta)):
    """
    Run up to 50 estimates in parallel. Per-item errors are captured in
    `error` field; the batch itself never fails with 5xx.
    """
    async def _one(item: PricingRequest) -> BatchItem:
        try:
            r = await _predict_one(item, key_meta["name"])
            return BatchItem(**r.model_dump())
        except HTTPException as e:
            return BatchItem(ok=False, job_id=item.job_id, error=e.detail)
        except Exception as e:
            return BatchItem(ok=False, job_id=item.job_id, error="Internal error")

    results = await asyncio.gather(*[_one(item) for item in req.estimates])
    return BatchPricingResponse(results=list(results))


@app.post("/.netlify/functions/pricing-outcome", response_model=OutcomeResponse)
async def record_outcome(req: OutcomeRequest,
                         key_meta: dict = Depends(_get_key_meta)):
    """
    Called by HouseAccount when a job closes with a final_price.
    Stores the label, computes APE against our midpoint, and schedules
    a background retrain when RETRAIN_THRESHOLD new outcomes accumulate.
    """
    ape = None
    try:
        with state.db_lock:
            with sqlite3.connect(DB_PATH) as conn:
                row = conn.execute(
                    "SELECT estimate_midpoint FROM estimates WHERE job_id = ?",
                    (req.job_id,),
                ).fetchone()
                if row and row[0] and row[0] > 0:
                    ape = round(abs(req.final_price - row[0]) / req.final_price * 100, 2)
                conn.execute(
                    "INSERT OR REPLACE INTO outcomes (job_id, final_price, ape, recorded_at)"
                    " VALUES (?,?,?,?)",
                    (req.job_id, req.final_price, ape,
                     datetime.now(timezone.utc).isoformat()),
                )
    except Exception as e:
        _log("db_error", op="record_outcome", job_id=req.job_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to store outcome")

    _log("outcome", job_id=req.job_id[:8], final_price=req.final_price,
         ape=ape, key_name=key_meta["name"])

    asyncio.create_task(_maybe_retrain())
    return OutcomeResponse(job_id=req.job_id, ape=ape)


@app.post("/retrain", dependencies=[Depends(_get_key_meta)])
async def trigger_retrain():
    """Manual retrain for ops. Non-blocking."""
    if state.retrain_running:
        return {"ok": True, "message": "Retrain already in progress"}
    state.retrain_running = True
    asyncio.get_event_loop().run_in_executor(None, _retrain_sync)
    return {"ok": True, "message": "Retrain started"}


@app.post("/keys/reload", dependencies=[Depends(_get_key_meta)])
async def reload_keys():
    """
    Hot-reload API keys from env without restart.
    Rotate keys by: add new key to API_KEYS → call this endpoint → remove old key.
    """
    n = state.key_store.reload()
    _log("keys_reloaded", count=n)
    return {"ok": True, "keys_loaded": n}


@app.get("/metrics")
def metrics():
    """Rolling accuracy stats from the outcome DB."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            total_estimates = conn.execute(
                "SELECT COUNT(*) FROM estimates"
            ).fetchone()[0]
            total_outcomes = conn.execute(
                "SELECT COUNT(*) FROM outcomes"
            ).fetchone()[0]
            mape_row = conn.execute("""
                SELECT AVG(ape), COUNT(*) FROM outcomes
                WHERE ape IS NOT NULL
                  AND recorded_at >= datetime('now', '-90 days')
            """).fetchone()
            per_cat = conn.execute("""
                SELECT e.service_category, AVG(o.ape), COUNT(*)
                FROM outcomes o JOIN estimates e ON o.job_id = e.job_id
                WHERE o.ape IS NOT NULL
                  AND o.recorded_at >= datetime('now', '-90 days')
                GROUP BY e.service_category
                ORDER BY AVG(o.ape)
            """).fetchall()
        cb = state.circuit_breaker
        return {
            "ok":              True,
            "total_estimates": total_estimates,
            "total_outcomes":  total_outcomes,
            "rolling_mape_90d": round(mape_row[0], 2) if mape_row[0] else None,
            "outcomes_90d":    mape_row[1],
            "n_train":         state.meta.get("n_train"),
            "model_version":   state.meta.get("model_version", "heila-v1.0.0"),
            "retrain_running": state.retrain_running,
            "retrain_threshold": RETRAIN_THRESHOLD,
            "circuit_breaker": {
                "open":     cb.is_open if cb else False,
                "failures": cb._failures if cb else 0,
            },
            "per_category_mape_90d": [
                {"category": r[0], "mape": round(r[1], 1), "n": r[2]}
                for r in per_cat
            ],
        }
    except Exception as e:
        _log("metrics_error", error=str(e))
        raise HTTPException(status_code=500, detail="Metrics unavailable")


@app.get("/health")
def health():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            n_outcomes = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
    except Exception:
        n_outcomes = -1
    cb = state.circuit_breaker
    return {
        "ok":              True,
        "models_loaded":   len(state.models),
        "model_version":   state.meta.get("model_version", "heila-v1.0.0"),
        "cache_size":      len(state.cache),
        "n_outcomes":      n_outcomes,
        "retrain_running": state.retrain_running,
        "circuit_open":    cb.is_open if cb else False,
    }
