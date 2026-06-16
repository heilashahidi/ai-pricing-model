"""
FastAPI pricing service.
Loads XGBoost models at startup, handles Haiku scope extraction + inference.
Records every estimate in SQLite; accepts final_price outcomes to close the
feedback loop and trigger background model retraining.

Observability stack:
  - Structured JSON logs on stdout (ship via Railway log drain)
  - Prometheus metrics at /metrics/prometheus (scrape with Grafana)
  - Strict health probe at /healthz (ping with Better Uptime / Cronitor)
  - Daily MAPE drift check with Slack webhook alert

Usage:
    python3 -m uvicorn pricing_service:app --port 8001
"""
import asyncio
import contextvars
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
import uuid
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
from starlette.middleware.base import BaseHTTPMiddleware

# Optional: prometheus_client (stub silently if missing so tests run without it)
try:
    from prometheus_client import (
        Counter, Gauge, Histogram,
        make_asgi_app as _make_prom_app,
    )
    _PROM = True
except ImportError:
    class _Noop:
        def labels(self, **_): return self
        def inc(self, *a): pass
        def observe(self, *a): pass
        def set(self, *a): pass
    def Counter(*a, **k):   return _Noop()
    def Gauge(*a, **k):     return _Noop()
    def Histogram(*a, **k): return _Noop()
    def _make_prom_app():   return None
    _PROM = False

# Optional: httpx for async Slack webhooks (available via anthropic dep)
try:
    import httpx as _httpx
except ImportError:
    _httpx = None

# ── State income table ─────────────────────────────────────────────────────

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

# ── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_SERVICE = os.environ.get("SERVICE_NAME", "pricing-ml")
_ENV     = os.environ.get("RAILWAY_ENVIRONMENT",
           os.environ.get("ENV", "development"))

# ContextVar lets each request coroutine carry its own request_id through
# every _log() call without threading-style explicit passing.
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def _log(event: str, **fields) -> None:
    log.info(json.dumps({
        "ts":         datetime.now(timezone.utc).isoformat(),
        "service":    _SERVICE,
        "env":        _ENV,
        "request_id": _request_id.get(),
        "event":      event,
        **fields,
    }))


# ── Constants ──────────────────────────────────────────────────────────────

MODELS_DIR        = "models"
META_PATH         = os.path.join(MODELS_DIR, "meta.json")
DB_PATH           = os.path.join(MODELS_DIR, "outcomes.db")
CACHE_PATH        = os.path.join(MODELS_DIR, "extraction_cache.pkl")
FEATURES_CSV      = "features_enriched.csv"
RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", "20"))
# Alert when rolling 30d MAPE exceeds this (default = 1.5 × 11.6% baseline)
DRIFT_THRESHOLD   = float(os.environ.get("DRIFT_MAPE_THRESHOLD", "17.4"))

EXTRACTION_TOOL = {
    "name": "extract_scope",
    "description": "Extract scope features from a home service job description.",
    "input_schema": {
        "type": "object",
        "properties": {
            "labor_only":      {"type": "boolean",
                                "description": "True only if the customer explicitly states they are supplying all parts or materials (e.g. 'we supply', 'I provide the materials', 'labor only'). Do NOT infer from service type — cleaning, landscaping, and moving are not labor-only unless the customer says so."},
            "task_count":      {"type": "integer",
                                "description": "Number of distinct task types. 'Install shutters + patch wall' = 2. 'Install 3 shutters' = 1 (one task type, multiple units)."},
            "unit_count":      {"type": "integer",
                                "description": "Number of physical items to work on. 'Install 3 shutters' = 3, 'replace 5 outlets' = 5, 'fix faucet' = 1. Counts items, not task types."},
            "complexity_tier": {"type": "string", "enum": ["low", "medium", "high"],
                                "description": "low=simple, medium=moderate, high=multi-trade."},
            "has_area_measure":{"type": "boolean",
                                "description": "True if an explicit area or size measurement is stated: square footage, acreage, or linear footage (e.g. '1,400 sq ft', '0.25 acres', '50 linear feet'). Bedroom or bathroom count alone (e.g. '3-bedroom home') does NOT qualify."},
        },
        "required": ["labor_only", "task_count", "unit_count", "complexity_tier", "has_area_measure"],
    },
}
EXTRACTION_DEFAULTS = {
    "labor_only": False, "task_count": 1, "unit_count": 1,
    "complexity_tier": "medium", "has_area_measure": False,
}

# ── Prometheus metrics ─────────────────────────────────────────────────────
# Exposed at /metrics/prometheus for Grafana to scrape.
# All category labels kept; key_name intentionally omitted (high-cardinality).

_REQ_DURATION = Histogram(
    "pricing_request_duration_seconds",
    "End-to-end estimate latency",
    ["category", "endpoint"],
    buckets=[.05, .1, .25, .5, 1.0, 2.0, 5.0],
)
_CONFIDENCE = Histogram(
    "pricing_confidence_score",
    "Model confidence distribution by category",
    ["category"],
    buckets=[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0],
)
_REQUESTS     = Counter("pricing_requests_total",
                         "Total estimate requests", ["category", "status"])
_LLM_CALLS    = Counter("pricing_llm_calls_total",
                         "Anthropic API calls made")
_LLM_FAILURES = Counter("pricing_llm_failures_total",
                         "Anthropic API call failures")
_CACHE_HITS   = Counter("pricing_scope_cache_hits_total",
                         "Scope extraction in-memory cache hits")
_OUTCOMES     = Counter("pricing_outcomes_total",
                         "Outcome records received")
_CIRCUIT_OPEN = Gauge("pricing_circuit_open",
                       "1 if Anthropic circuit breaker is open")
_N_TRAIN      = Gauge("pricing_n_train",
                       "Training rows in current live model")
_MAPE_90D     = Gauge("pricing_mape_90d",
                       "Rolling 90-day MAPE by category", ["category"])
_RETRAIN_RUNNING = Gauge("pricing_retrain_running",
                          "1 if a background retrain is in progress")


# ── Reliability primitives ─────────────────────────────────────────────────

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
            self._opened_at = None
            return False
        return True

    def on_success(self) -> None:
        self._failures  = 0
        self._opened_at = None
        _CIRCUIT_OPEN.set(0)

    def on_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold and self._opened_at is None:
            self._opened_at = time.monotonic()
            _CIRCUIT_OPEN.set(1)
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
    Multi-tenant API key registry loaded from env at startup.

    Single key (backward compat):
        GAUNTLET_PRICING_SECRET=<hex>

    Multi-tenant:
        API_KEYS=[{"key":"<hex>","name":"prod","rate_limit":600}, ...]

    Hot-reload: POST /keys/reload (re-reads env without restart).
    Rotation: add new key → reload → remove old key.
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
    models:          dict   = {}
    meta:            dict   = {}
    client:          object = None
    cache:           LRUCache = LRUCache(maxsize=2048)
    nomi:            object = None
    db_lock:         object = None
    retrain_running: bool   = False
    circuit_breaker: object = None
    rate_limiter:    object = None
    key_store:       object = None

state = State()


# ── Middleware: request ID ─────────────────────────────────────────────────

class _RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Stamps every request with a short random ID.
    Reads X-Request-ID header if provided (caller correlation); otherwise mints one.
    The ID flows into every _log() call via _request_id ContextVar.
    When you ship logs to Datadog/Loki you can search by request_id to see
    the full trace for a single request across all log lines.
    """
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        _request_id.set(rid)
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


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

            CREATE TABLE IF NOT EXISTS model_health (
                checked_at    TEXT NOT NULL,
                mape_30d      REAL,
                n_outcomes    INTEGER,
                baseline_mape REAL,
                drift_ratio   REAL,
                alert_sent    INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_outcomes_recorded
                ON outcomes(recorded_at);
        """)


def _lookup_stored_estimate(job_id: str) -> Optional["PricingResponse"]:
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

    # Load training benchmarks from eval_results.json
    if os.path.exists("eval_results.json"):
        with open("eval_results.json") as f:
            _er = json.load(f)
        state.meta.setdefault(
            "baseline_mape",
            _er.get("benchmark", {}).get("blended_baseline_mape", 11.6),
        )
        state.meta["training_benchmarks"] = {
            "rmse_baseline":          _er.get("baseline",          {}).get("RMSE"),
            "rmse_routed":            _er.get("routed",             {}).get("RMSE"),
            "wape_baseline":          _er.get("baseline",          {}).get("WAPE"),
            "wape_routed":            _er.get("routed",             {}).get("WAPE"),
            "mape_baseline":          _er.get("baseline",          {}).get("MAPE"),
            "mape_routed":            _er.get("routed",             {}).get("MAPE"),
            "bias_baseline":          _er.get("baseline",          {}).get("Bias"),
            "bias_routed":            _er.get("routed",             {}).get("Bias"),
            "handyman_rmse_baseline": _er.get("handyman_baseline", {}).get("RMSE"),
            "handyman_rmse_model":    _er.get("handyman_model",    {}).get("RMSE"),
            "handyman_wape_baseline": _er.get("handyman_baseline", {}).get("WAPE"),
            "handyman_wape_model":    _er.get("handyman_model",    {}).get("WAPE"),
            "handyman_mape_baseline": _er.get("handyman_baseline", {}).get("MAPE"),
            "handyman_mape_model":    _er.get("handyman_model",    {}).get("MAPE"),
        }
    else:
        state.meta.setdefault("baseline_mape", 11.6)
        state.meta.setdefault("training_benchmarks", {})

    # Seed model_version if not present in meta.json (first run after adding this field)
    state.meta.setdefault("model_version", "heila-v1.0.0")

    _log("models_loaded", n_train=state.meta["n_train"],
         version=state.meta["model_version"],
         baseline_mape=state.meta["baseline_mape"])
    _N_TRAIN.set(state.meta["n_train"])

    # Extraction cache: restore from disk
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                state.cache.update(pickle.load(f))
            _log("cache_restored", size=len(state.cache))
        except Exception as e:
            _log("cache_restore_failed", error=str(e))

    # pgeocode + ZIP prewarm
    _log("startup", phase="pgeocode")
    state.nomi = pgeocode.Nominatim("us")
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

    # Auth secret — fail fast so a misconfigured deploy never silently accepts
    # all requests (empty key store passes nothing through, but silently).
    if not os.environ.get("GAUNTLET_PRICING_SECRET") and not os.environ.get("API_KEYS"):
        raise RuntimeError("GAUNTLET_PRICING_SECRET (or API_KEYS) must be set")

    # Reliability + auth primitives
    state.circuit_breaker = _CircuitBreaker(threshold=5, timeout=60.0)
    state.rate_limiter    = _RateLimiter()
    state.key_store       = _ApiKeyStore()

    # DB
    state.db_lock = threading.Lock()
    _init_db(DB_PATH)

    # Prometheus initial gauge sync
    await _update_prom_gauges()

    # Start daily health-check loop
    health_task = asyncio.create_task(_health_check_loop())

    _log("startup", phase="ready", prom_available=_PROM)

    yield

    # Clean shutdown
    health_task.cancel()
    try:
        await health_task
    except asyncio.CancelledError:
        pass

    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(dict(state.cache), f)
        _log("cache_persisted", size=len(state.cache))
    except Exception as e:
        _log("cache_persist_failed", error=str(e))

    _log("shutdown")


app = FastAPI(title="HouseAccount Pricing Service", lifespan=lifespan)
app.add_middleware(_RequestIDMiddleware)

# Mount Prometheus scrape endpoint (Grafana → Connections → Data Sources → Prometheus)
_prom_app = _make_prom_app()
if _prom_app:
    app.mount("/metrics/prometheus", _prom_app)


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
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization[len("Bearer "):]

    ks = state.key_store
    if ks is None:
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
    scope:             Optional[dict] = None

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
    key = hashlib.sha256(description.encode()).hexdigest()
    if key in state.cache:
        _CACHE_HITS.inc()
        return state.cache[key]

    cb = state.circuit_breaker
    if cb and cb.is_open:
        return EXTRACTION_DEFAULTS

    _LLM_CALLS.inc()
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
            _LLM_FAILURES.inc()
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
        orig_est = medians.get(_resolve_category(req.service_category), medians.get("overall", 300.0))

    resolved = _resolve_category(req.service_category)
    cat_vec = [1 if resolved == c else 0 for c in categories]
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
    if (hi - lo) > 3 * meta.get("training_median_interval", 212):
        cap = min(cap, 0.45)
    if _resolve_category(category) not in set(meta.get("production_categories", [])):
        cap = min(cap, 0.40)

    return round(min(base, cap), 4)


# Categories where the model has high uncertainty due to small labeled datasets.
# Their prediction intervals are too tight (coverage dropped from 97.3% baseline
# to ~93%), causing actual prices to fall outside the lo/hi range too often.
_HIGH_VARIANCE_CATEGORIES = {"Handyman", "Plumbing", "Electrical", "Flooring"}

# Maps PRD production-vertical slugs (lowercase-hyphenated, per Appendix A) to
# the Title Case category names used in training data.  Lets clients that follow
# the PRD's canonical format ("electrical", "pest-control", etc.) resolve to the
# correct training category instead of being capped as unknown/OOD.
_CATEGORY_ALIASES: dict = {
    "electrical":              "Electrical",
    "exterior-cleaning":       "Exterior",
    "handyman":                "Handyman",
    "hvac":                    "HVAC",
    "indoor-cleaning":         "Cleaning",
    "irrigation":              "Landscaping",
    "landscaping-lawn":        "Landscaping",
    "pest-control":            "Pest Control",
    "plumbing":                "Plumbing",
    "tick-mosquito-treatment": "Pest Control",
}


def _resolve_category(raw: str) -> str:
    """Normalize a category string to its training Title Case form.

    Accepts both PRD kebab-case slugs (e.g. 'pest-control') and the Title Case
    strings used in training data (e.g. 'Pest Control').  Unknown values pass
    through unchanged so OOD detection still fires on truly unknown categories.
    """
    normalized = raw.strip().lower().replace(" ", "-")
    return _CATEGORY_ALIASES.get(normalized, raw)


def route_predict(req: PricingRequest, X: np.ndarray) -> tuple[float, float, float]:
    well_priced  = set(state.meta.get("well_priced_categories", []))
    use_baseline = _resolve_category(req.service_category) in well_priced and req.original_estimate

    if use_baseline:
        lo  = req.original_estimate_lo  if req.original_estimate_lo  is not None else req.original_estimate * 0.8
        hi  = req.original_estimate_hi  if req.original_estimate_hi  is not None else req.original_estimate * 1.2
        return fix_intervals(lo, req.original_estimate, hi)

    lo, mid, hi = fix_intervals(
        float(state.models[0.05].predict(X)[0]),
        float(state.models[0.5].predict(X)[0]),
        float(state.models[0.95].predict(X)[0]),
    )

    # Widen intervals for high-variance categories to restore coverage.
    # The CF model produces tighter intervals than the true uncertainty warrants
    # for categories with <20 labeled rows. 25% symmetric widening recovers
    # the coverage gap (97.3% baseline → 93.4% model) without changing the midpoint.
    if _resolve_category(req.service_category) in _HIGH_VARIANCE_CATEGORIES:
        spread = (hi - lo) * 0.25
        lo -= spread
        hi += spread

    return lo, mid, hi


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
        "objective": "reg:quantileerror", "quantile_alpha": quantile,
        "n_estimators": 400, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
        "random_state": 42, "verbosity": 0,
    }


def _retrain_sync() -> None:
    """Blocking retrain — runs in thread-pool executor."""
    try:
        meta           = state.meta
        deadline_map   = meta["deadline_map"]
        complexity_map = meta["complexity_map"]
        categories     = meta["categories"]

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
        _log("retrain_start", csv_rows=len(csv_rows), db_rows=len(db_rows), total=len(y))

        new_models = {q: xgb.XGBRegressor(**_xgb_params(q)) for q in [0.05, 0.5, 0.95]}
        for q, m in new_models.items():
            m.fit(X, y)

        tag_map = {0.05: "q005", 0.5: "q050", 0.95: "q095"}
        for q, m in new_models.items():
            tag = tag_map[q]
            tmp = os.path.join(MODELS_DIR, f"_tmp_xgb_{tag}.joblib")
            dst = os.path.join(MODELS_DIR, f"xgb_{tag}.joblib")
            joblib.dump(m, tmp)
            os.rename(tmp, dst)

        state.models.update(new_models)
        new_version             = _bump_version(state.meta.get("model_version", "heila-v1.0.0"))
        state.meta["n_train"]       = len(y)
        state.meta["model_version"] = new_version
        with open(META_PATH, "w") as f:
            json.dump(state.meta, f, indent=2)

        _N_TRAIN.set(len(y))
        _log("retrain_complete", n_train=len(y), new_version=new_version)

    except Exception as e:
        _log("retrain_error", error=str(e))
    finally:
        state.retrain_running = False
        _RETRAIN_RUNNING.set(0)


async def _maybe_retrain() -> None:
    if state.retrain_running:
        return
    try:
        with sqlite3.connect(DB_PATH) as conn:
            new_labeled = conn.execute(
                "SELECT COUNT(*) FROM outcomes o JOIN estimates e ON o.job_id = e.job_id"
            ).fetchone()[0]
        if new_labeled >= RETRAIN_THRESHOLD and new_labeled > state.meta.get("n_train", 0):
            _log("retrain_triggered", new_labeled=new_labeled)
            state.retrain_running = True
            _RETRAIN_RUNNING.set(1)
            asyncio.get_event_loop().run_in_executor(None, _retrain_sync)
    except Exception as e:
        _log("retrain_check_error", error=str(e))


# ── Observability: Prometheus gauge sync + MAPE drift alert ───────────────

async def _update_prom_gauges() -> None:
    """Sync Prometheus gauges from DB. Called at startup and after each health check."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("""
                SELECT e.service_category, AVG(o.ape)
                FROM outcomes o JOIN estimates e ON o.job_id = e.job_id
                WHERE o.ape IS NOT NULL
                  AND o.recorded_at >= datetime('now', '-90 days')
                GROUP BY e.service_category
            """).fetchall()
        for cat, mape in rows:
            if mape is not None:
                _MAPE_90D.labels(category=cat).set(mape)
    except Exception:
        pass
    cb = state.circuit_breaker
    _CIRCUIT_OPEN.set(1 if (cb and cb.is_open) else 0)
    _N_TRAIN.set(state.meta.get("n_train", 0))
    _RETRAIN_RUNNING.set(1 if state.retrain_running else 0)


async def _send_drift_alert(mape_30d: float, n: int, ratio: float) -> None:
    """POST a Slack webhook when MAPE drift exceeds threshold."""
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        _log("drift_alert_no_webhook", mape_30d=mape_30d, ratio=ratio)
        return

    baseline = state.meta.get("baseline_mape", 11.6)
    payload  = {
        "text": (
            f":warning: *HouseAccount Pricing — MAPE Drift Alert*\n"
            f"30-day MAPE: *{mape_30d:.1f}%* vs baseline {baseline:.1f}% "
            f"({ratio:.1f}x — threshold {DRIFT_THRESHOLD:.1f}%)\n"
            f"Based on {n} outcomes over the last 30 days.\n"
            f"Check `/metrics` for per-category breakdown. "
            f"Trigger `POST /retrain` if retraining data is sufficient."
        )
    }

    try:
        if _httpx:
            async with _httpx.AsyncClient() as client:
                r = await client.post(webhook, json=payload, timeout=10.0)
                _log("drift_alert_sent", http_status=r.status_code,
                     mape_30d=mape_30d, ratio=ratio)
        else:
            import urllib.request as _ur
            req = _ur.Request(webhook, data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
            _ur.urlopen(req, timeout=10)
            _log("drift_alert_sent", mape_30d=mape_30d, ratio=ratio)
    except Exception as e:
        _log("drift_alert_failed", error=str(e))


async def _check_mape_drift() -> None:
    """
    Compute 30-day rolling MAPE. If it exceeds DRIFT_THRESHOLD, fire a
    Slack alert and write a record to model_health for audit trail.
    Requires at least 10 outcomes to avoid false positives on thin data.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("""
                SELECT AVG(ape), COUNT(*)
                FROM outcomes
                WHERE ape IS NOT NULL
                  AND recorded_at >= datetime('now', '-30 days')
            """).fetchone()

        mape_30d, n = row[0], row[1]

        if mape_30d is None or n < 10:
            _log("drift_check_skipped", n=n, reason="insufficient_data")
            return

        baseline   = state.meta.get("baseline_mape", 11.6)
        ratio      = round(mape_30d / baseline, 2)
        alert_sent = 0

        _log("drift_check", mape_30d=round(mape_30d, 1),
             baseline=baseline, ratio=ratio, n=n, threshold=DRIFT_THRESHOLD)

        if mape_30d > DRIFT_THRESHOLD:
            await _send_drift_alert(round(mape_30d, 1), n, ratio)
            alert_sent = 1

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """INSERT INTO model_health
                   (checked_at, mape_30d, n_outcomes, baseline_mape, drift_ratio, alert_sent)
                   VALUES (?,?,?,?,?,?)""",
                (datetime.now(timezone.utc).isoformat(),
                 round(mape_30d, 2), n, baseline, ratio, alert_sent),
            )
    except Exception as e:
        _log("drift_check_error", error=str(e))


async def _health_check_loop() -> None:
    """
    Background coroutine started in lifespan. Runs drift check + gauge sync
    daily. Initial delay of 1 hour gives the DB time to accumulate outcomes
    before the first check.
    """
    await asyncio.sleep(3600)
    while True:
        await _check_mape_drift()
        await _update_prom_gauges()
        await asyncio.sleep(86400)


# ── Core inference ─────────────────────────────────────────────────────────

async def _predict_one(req: PricingRequest, key_name: str) -> PricingResponse:
    t0 = time.monotonic()

    stored = _lookup_stored_estimate(req.job_id)
    if stored:
        latency = time.monotonic() - t0
        _REQ_DURATION.labels(category=req.service_category, endpoint="estimate").observe(latency)
        _REQUESTS.labels(category=req.service_category, status="idempotent").inc()
        _log("estimate_idempotent", job_id=req.job_id[:8],
             category=req.service_category, key_name=key_name,
             latency_ms=round(latency * 1000))
        return stored

    desc_hash    = hashlib.sha256(req.job_description.encode()).hexdigest()
    scope_cached = desc_hash in state.cache

    scope       = await extract_scope(req.job_description)
    X           = build_feature_vector(req, scope)
    lo, mid, hi = route_predict(req, X)

    lo  = max(lo,  0.0)
    mid = max(mid, 1.0)
    hi  = max(hi,  mid)

    confidence    = compute_confidence(lo, hi, mid, req.service_category)
    model_version = state.meta.get("model_version", "heila-v1.0.0")
    latency       = time.monotonic() - t0

    _REQ_DURATION.labels(category=req.service_category, endpoint="estimate").observe(latency)
    _CONFIDENCE.labels(category=req.service_category).observe(confidence)
    _REQUESTS.labels(category=req.service_category, status="success").inc()

    _log("estimate",
         job_id=req.job_id[:8], category=req.service_category, zip=req.zip_code,
         lo=round(lo), mid=round(mid), hi=round(hi), confidence=confidence,
         latency_ms=round(latency * 1000), model_version=model_version,
         scope_cached=scope_cached, key_name=key_name,
         labor_only=scope.get("labor_only"),
         tasks=scope.get("task_count"),
         units=scope.get("unit_count"))

    _store_estimate(req.job_id, req.service_category, req.zip_code,
                    round(lo, 2), round(hi, 2), round(mid, 2),
                    confidence, model_version, X)

    return PricingResponse(
        job_id=req.job_id,
        estimate_lo=round(lo, 2), estimate_hi=round(hi, 2),
        estimate_midpoint=round(mid, 2), confidence=confidence,
        model_version=model_version,
        scope={
            "labor_only":      scope.get("labor_only", False),
            "task_count":      scope.get("task_count", 1),
            "unit_count":      scope.get("unit_count", 1),
            "complexity_tier": scope.get("complexity_tier", "medium"),
            "has_area_measure":scope.get("has_area_measure", False),
        },
    )


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/.netlify/functions/pricing-estimate", response_model=PricingResponse)
async def predict(req: PricingRequest, key_meta: dict = Depends(_get_key_meta)):
    return await _predict_one(req, key_meta["name"])


@app.post("/.netlify/functions/pricing-estimate-batch",
          response_model=BatchPricingResponse)
async def predict_batch(req: BatchPricingRequest,
                        key_meta: dict = Depends(_get_key_meta)):
    async def _one(item: PricingRequest) -> BatchItem:
        try:
            r = await _predict_one(item, key_meta["name"])
            return BatchItem(**r.model_dump())
        except HTTPException as e:
            return BatchItem(ok=False, job_id=item.job_id, error=e.detail)
        except Exception:
            return BatchItem(ok=False, job_id=item.job_id, error="Internal error")

    results = await asyncio.gather(*[_one(item) for item in req.estimates])
    return BatchPricingResponse(results=list(results))


@app.post("/.netlify/functions/pricing-outcome", response_model=OutcomeResponse)
async def record_outcome(req: OutcomeRequest,
                         key_meta: dict = Depends(_get_key_meta)):
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

    _OUTCOMES.inc()
    _log("outcome", job_id=req.job_id[:8], final_price=req.final_price,
         ape=ape, key_name=key_meta["name"])

    asyncio.create_task(_maybe_retrain())
    return OutcomeResponse(job_id=req.job_id, ape=ape)


@app.post("/retrain", dependencies=[Depends(_get_key_meta)])
async def trigger_retrain():
    if state.retrain_running:
        return {"ok": True, "message": "Retrain already in progress"}
    state.retrain_running = True
    _RETRAIN_RUNNING.set(1)
    asyncio.get_event_loop().run_in_executor(None, _retrain_sync)
    return {"ok": True, "message": "Retrain started"}


@app.post("/keys/reload", dependencies=[Depends(_get_key_meta)])
async def reload_keys():
    n = state.key_store.reload()
    _log("keys_reloaded", count=n)
    return {"ok": True, "keys_loaded": n}


@app.get("/healthz")
def healthz():
    """
    Strict probe for uptime monitors (Better Uptime, Cronitor, UptimeRobot).
    Returns 200 only when fully operational; 503 otherwise.
    Configure your monitor to alert on any non-200 response.

    What triggers 503:
      - Models not loaded (startup not complete)
      - Anthropic circuit breaker open (LLM unreachable for >5 consecutive calls)
      - SQLite DB unreachable
    """
    issues = []
    if len(state.models) < 3:
        issues.append("models_not_loaded")
    cb = state.circuit_breaker
    if cb and cb.is_open:
        issues.append("anthropic_circuit_open")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("SELECT 1 FROM estimates LIMIT 1")
    except Exception:
        issues.append("db_unreachable")

    if issues:
        return JSONResponse(status_code=503,
                            content={"ok": False, "issues": issues})
    return {
        "ok":            True,
        "model_version": state.meta.get("model_version", "unknown"),
    }


@app.get("/metrics")
def metrics():
    """Human-readable JSON metrics. For Prometheus scraping use /metrics/prometheus."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            total_estimates = conn.execute(
                "SELECT COUNT(*) FROM estimates"
            ).fetchone()[0]
            total_outcomes = conn.execute(
                "SELECT COUNT(*) FROM outcomes"
            ).fetchone()[0]
            summary_row = conn.execute("""
                SELECT
                  AVG(o.ape),
                  SQRT(AVG((o.ape / 100.0 * o.final_price) * (o.ape / 100.0 * o.final_price))),
                  SUM(o.ape / 100.0 * o.final_price) / NULLIF(SUM(o.final_price), 0) * 100,
                  COUNT(*)
                FROM outcomes o
                WHERE o.ape IS NOT NULL
                  AND o.recorded_at >= datetime('now', '-90 days')
            """).fetchone()
            per_cat = conn.execute("""
                SELECT
                  e.service_category,
                  AVG(o.ape),
                  SQRT(AVG((o.ape / 100.0 * o.final_price) * (o.ape / 100.0 * o.final_price))),
                  SUM(o.ape / 100.0 * o.final_price) / NULLIF(SUM(o.final_price), 0) * 100,
                  COUNT(*)
                FROM outcomes o JOIN estimates e ON o.job_id = e.job_id
                WHERE o.ape IS NOT NULL
                  AND o.recorded_at >= datetime('now', '-90 days')
                GROUP BY e.service_category ORDER BY AVG(o.ape)
            """).fetchall()
            last_health = conn.execute(
                "SELECT checked_at, mape_30d, drift_ratio, alert_sent "
                "FROM model_health ORDER BY checked_at DESC LIMIT 1"
            ).fetchone()

        cb = state.circuit_breaker
        rolling_mape = round(summary_row[0], 2) if summary_row[0] else None
        rolling_rmse = round(summary_row[1], 2) if summary_row[1] else None
        rolling_wape = round(summary_row[2], 2) if summary_row[2] else None
        outcomes_90d = summary_row[3]
        return {
            "ok":              True,
            "total_estimates": total_estimates,
            "total_outcomes":  total_outcomes,
            "n_train":         state.meta.get("n_train"),
            "model_version":   state.meta.get("model_version", "heila-v1.0.0"),
            "retrain_running": state.retrain_running,
            "retrain_threshold": RETRAIN_THRESHOLD,
            "drift_threshold": DRIFT_THRESHOLD,
            "circuit_breaker": {
                "open":     cb.is_open if cb else False,
                "failures": cb._failures if cb else 0,
            },
            "rolling_90d": {
                "mape": rolling_mape,
                "rmse": rolling_rmse,
                "wape": rolling_wape,
                "n":    outcomes_90d,
            },
            "training_benchmarks": state.meta.get("training_benchmarks", {}),
            "last_health_check": {
                "checked_at":  last_health[0],
                "mape_30d":    last_health[1],
                "drift_ratio": last_health[2],
                "alert_sent":  bool(last_health[3]),
            } if last_health else None,
            "per_category_90d": [
                {
                    "category": r[0],
                    "mape": round(r[1], 1) if r[1] else None,
                    "rmse": round(r[2], 2) if r[2] else None,
                    "wape": round(r[3], 2) if r[3] else None,
                    "n":    r[4],
                }
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
