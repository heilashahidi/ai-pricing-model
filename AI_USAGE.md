# AI Usage Log

## Tools used

### Claude Code (Claude Sonnet 4.6) — primary coding agent
Used throughout the entire project via the Claude Code CLI. Role: architecture design, data analysis, all code generation, debugging, and documentation. This was not "generate code and paste" — Claude Code had access to the filesystem and ran commands directly, so it could read the CSV, measure the baseline failures, iterate on the model, and verify results against the actual HouseAccount staging API in a single session.

### Claude Haiku (`claude-haiku-4-5-20251001`) — scope feature extractor
Used at two points: (1) batch offline extraction on all 1,432 job descriptions to build training features, and (2) online at inference time to extract scope signals from incoming job descriptions. Called via the Anthropic API with structured tool-use mode to guarantee valid JSON output. Approximate cost: ~$0.60 for the full batch extraction run.

---

## Significant prompts

These are the prompts and decisions that shaped the architecture — not every line of generated code.

### 1. Finding where the baseline actually fails

Before writing any code I asked Claude Code to analyze the CSV and compute per-category MAPE against the `original_estimate` baseline. The output was decisive:

> *"Handyman: 48.4% MAPE on 14 priced rows. Every one of the 10 worst predictions is a Handyman job. The errors are scope-driven: 'install supplied exterior shutters' was estimated at $750, actual was $250 — a labor-only job priced as if materials were included."*

This changed the entire approach. The problem wasn't model weakness — it was that the baseline was blind to job description text. Every architectural decision followed from that finding.

### 2. Defining the Haiku extraction schema

The extraction prompt was iterated to cover the cases that cause the most MAPE damage:

```
Extract scope features from this home service job description.
labor_only=true if the customer supplies all materials/parts.
task_count=number of distinct tasks.
complexity_tier: low/medium/high.
has_area_measure=true if sq ft, rooms, or linear footage mentioned.
```

The key addition was the `labor_only` definition: "true if the customer supplies all materials/parts" with examples ("we supply," "you supply," assembly of items the customer already owns). Without explicit examples, Haiku was missing implicit labor-only cases like "assemble furniture piece at home" — which is clearly labor-only but contains no "supply" keyword.

### 3. Validating extraction before training

Before training on the extracted features I ran `validate_extraction.py` on the 14 priced Handyman rows. The first run showed 9/14 accuracy — but on inspection, Haiku was correct on all 14 and my ground truth check was wrong. The script was checking for "we supply" keywords, missing implicit cases like "move plants" (pure labor) or "assemble Walmart shelving" (customer owns the items).

The lesson: validate your validator. The corrected run showed 14/14, which gave confidence to proceed with full extraction.

### 4. The routing strategy — when blended MAPE wouldn't budge

After training, the standalone XGBoost model had 14.0% blended MAPE — worse than the 11.6% baseline. The model was improving Handyman (48.4% → 31.4%) but adding noise to HVAC, Landscaping, and Cleaning, which are 80% of the priced rows.

Three approaches were tried in sequence:
- Two-tier blend (80% original / 20% model for well-priced categories): tied at 11.6%
- Correction-factor model (predict `final_price / original_estimate` instead of price directly): 11.7%, slightly worse
- Routing (use original_estimate for well-priced categories, model for hard ones): **11.4%**, beats baseline

The routing approach works because category is always in the request payload, so it's fully deployable. The core insight: for well-priced categories, `original_estimate` is already the best predictor — don't try to improve what isn't broken.

### 5. HMAC signing discovery

The HouseAccount staging API uses HMAC-SHA256 with a custom header scheme. The API reference said "hexdigest of timestamp + payload concatenated with period" — a standard enough description. The first implementation failed with 401.

Rather than guessing, Claude Code wrote a probe script that tried four variants systematically:
- Decoded secret + period separator + compact JSON → 401
- Decoded secret + period separator + spaced JSON → 401
- Decoded secret + no separator → 401
- **Raw base64 string as key + period separator + compact JSON → 201**

The API was treating the base64 string itself as the HMAC key, not the decoded bytes. This is non-standard and not documented. The variant probe found it in one run.

### 6. OOD confidence calibration

The three OOD conditions from the PRD needed specific thresholds:

```python
# midpoint > $5k (95th percentile of training data)
if mid > 5000: cap = min(cap, 0.40)

# interval > 3x median training interval ($212)
if (hi - lo) > 3 * 212: cap = min(cap, 0.45)

# category outside 8 production verticals
if category not in PRODUCTION_CATEGORIES: cap = min(cap, 0.40)
```

The base confidence formula `1 / (1 + interval_ratio)` needed a zero-guard (`if mid <= 0: return 0.3`) that the adversarial spec reviewer caught. The reviewer also caught that the "Exterior" dataset category was ambiguously mapped to production verticals — it maps to `exterior-cleaning` for cleaning jobs but should be OOD for painting or siding work.

### 7. Rails auth using `secure_compare`

The PRD references Node.js `timingSafeEqual` for auth. The Rails equivalent is `ActiveSupport::SecurityUtils.secure_compare`. Using a naive string comparison (`==`) for bearer token validation is a timing attack — an attacker can measure response time to guess the token character by character. `secure_compare` runs in constant time regardless of where the strings diverge.

### 8. PRD strict audit — catching 7 API contract failures

A parallel multi-agent audit of the codebase against the PRD's Appendix A found 7 blocking failures that would have caused an evaluator to fail the submission on first contact:

- **Endpoint path**: `/predict` → `/.netlify/functions/pricing-estimate`
- **Auth header**: `X-Internal-Key` (custom) → `Authorization: Bearer GAUNTLET_PRICING_SECRET`
- **Auth env var**: `PRICING_SERVICE_INTERNAL_KEY` → `GAUNTLET_PRICING_SECRET`
- **Error response key**: FastAPI's default `"detail"` → PRD-required `"error"` across all 5 error shapes (400 malformed JSON, 400 missing field, 401, 405, 500)
- **Validation errors**: FastAPI's default 422 → 400 with `{"error": "<field> required"}`

None of these were caught by the existing test suite because the Rails controller (the external-facing layer) already had the correct auth and error shapes — the failures were in the Python FastAPI service that the evaluators would also call directly. The fix required custom exception handlers in FastAPI and consolidating the two secrets into one.

### 9. Two inference bugs found by root cause analysis

A systematic investigation into why the model underperformed the baseline in 9 of 10 per-category MAPE comparisons found two inference bugs:

**Bug 1 — `original_estimate=None` fed 0 to the model's top feature.** When `original_estimate` is absent, the code did `orig_est = req.original_estimate or 0.0`. Zero is out-of-distribution for every training row and breaks XGBoost's most important feature (importance 0.128). Fix: store per-category median estimates in `meta.json` during training and use them as the fallback.

**Bug 2 — `state_median_income` hardcoded to national median at inference.** Training used real per-ZIP income from pgeocode. Serving used `75149 / 100_000` for every job regardless of location. Fix: load `pgeocode.Nominatim("us")` at startup and resolve the actual state income per ZIP on each request.

The investigation also clarified that the "9 of 10 categories worse" headline was a measurement artifact — 6 of those categories are `WELL_PRICED` and the routing strategy uses the baseline for them anyway. Actual deployed MAPE was already 11.41% vs 11.56% baseline.

---

## Validation steps for AI-generated code

**Haiku extraction accuracy** — `validate_extraction.py`
Manual review of all 14 priced Handyman rows. Checked `labor_only` against the actual job description text. 14/14 correct after fixing the ground truth check (see Prompt 3 above).

**Model evaluation** — `train.py` output
5-fold stratified cross-validation across all 411 priced rows. LOO-CV for Handyman specifically (14 rows is too few for a train/test split). Both benchmarks verified before moving to the serving layer.

**Rails controller** — `bundle exec rails test`
12 tests covering: missing auth (401), wrong token (401), malformed JSON (400), each missing required field (400), successful response shape, FastAPI timeout (500), FastAPI connection refused (500), GET request (405), OOD confidence passthrough. All use WebMock to isolate the controller from the actual FastAPI service.

**Integration test** — `integration_test.py`
15 tests running against the live stack: sections covering auth/validation, prediction quality (including OOD confidence caps and HVAC baseline passthrough), response time (cold-start 1140ms, warm avg 289ms, warm max 320ms — all under 2s), and three live bookings posted to HouseAccount staging (HTTP 201 each). All 15 passing against the Railway production deployment.

**Hallucinations / wrong output caught:**

1. **validate_extraction.py ground truth** — the initial script checked for literal "we supply" / "you supply" text and missed all implicit labor-only cases. Caught by reading the "WRONG" outputs and noticing Haiku was actually right.

2. **HMAC signing** — first implementation used `base64.b64decode(secret)` as the key. The actual API required the raw base64 string as bytes. Caught by the probe script returning 401 on all standard variants.

3. **Duplicate confidence section** — the adversarial spec reviewer (run during the /office-hours design session) flagged a duplicate `Confidence computation` block that was introduced during an edit. Caught before the design doc was approved.

4. **Staging test phone number reuse** — the integration test initially used hardcoded phone numbers. The second run returned HTTP 500 (HouseAccount staging rejected a duplicate booking). Fixed by appending a timestamp suffix to phone numbers per run.

5. **`estimate_midpoint` as `(lo + hi) / 2`** — the spec reviewer flagged this as explicitly wrong per the PRD ("Computing `(lo + hi) / 2` server-side assumes a uniform distribution and produces worse MAPE comparisons"). Fixed to use the q=0.5 quantile model directly.

6. **Wrong auth header on Python service** — the PRD audit found the Python FastAPI service used a custom `X-Internal-Key` header while the PRD mandates `Authorization: Bearer`. The Rails controller (the external-facing layer) already had the correct header, which is why the Rails tests passed but the Python service would fail when called directly by an evaluator.

7. **FastAPI error response key `"detail"` vs `"error"`** — FastAPI's default exception handlers return `{"detail": "..."}`. The PRD requires `{"error": "..."}` across all error codes. Also, FastAPI returns 422 for validation errors; the PRD requires 400. Both required custom exception handlers. Caught only by auditing the Python service directly against the PRD contract.

8. **Railway port mismatch** — `PRICING_SERVICE_URL` was set to port 8000 but the `pricing-ml` service runs on port 8080 (Railway assigns `PORT=8080`). The service was healthy but unreachable because all calls went to the wrong port. Caught by reading Railway logs via the CLI after the integration test returned consistent 500s.

---

## Reflection

**Where AI helped most:**

The data analysis step was the highest-leverage use. Within minutes of having the CSV, Claude Code identified the exact failure mode (Handyman scope-blindness), computed per-category MAPE, and traced the worst predictions to specific description patterns. This would have taken hours of EDA in a notebook and might have led to less targeted conclusions.

The HMAC signing debug was also faster than it would have been manually. Writing a probe script to try four variants and surface the 201 took about two minutes. Debugging this by reading documentation or trial-and-error could have taken much longer.

**Where it produced bad or incorrect output:**

The correction-factor model was a promising idea (predict `final/original` ratio instead of price) that didn't improve results. It was worth trying — the idea is sound for this kind of problem — but it added an iteration cycle without payoff.

The initial confidence formula lacked the zero-midpoint guard. That's a real edge case that causes a divide-by-zero crash, and it wasn't in the first draft. It took an adversarial code reviewer to surface it. Production code needs explicit edge case review beyond "does it work on the happy path."

**What I'd do differently:**

Start the HMAC probe earlier. I spent time reading the API reference carefully before discovering it was underspecified. Testing against the actual endpoint from the start (with a minimal payload) would have found the signing issue before building the full integration.

Also: the Handyman confidence intervals are too narrow (57% coverage vs. 80% target). The model is reasonably accurate on the point estimate but the q=0.1/q=0.9 quantile spread is too tight. With more time, I'd widen the quantile targets (q=0.05 / q=0.95) or add a post-hoc interval expansion for hard categories. This doesn't affect the MAPE benchmarks but would improve the calibration story for production routing.
