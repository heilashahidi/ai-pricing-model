# Gauntlet Project Brief: HouseAccount AI Pricing Model

## Step 1: Project Category

**Category B: AI Solution Building**

We are building an AI-powered pricing model that integrates with the booking system. This is a product, not a workflow automation. Category B tests whether candidates can architect and ship an AI feature, which is what the AI Engineer role demands.

---

## Step 2: Project Brief

### Project Title

HouseAccount AI Pricing Model: Estimates Homeowners and Providers Trust

### Problem Statement

Imagine a pricing engine so reliable that homeowners book without shopping the market and providers accept without renegotiation. That's the bar we're raising the model to. We want a system that combines internal pricing data with public knowledge bases to produce transparent booking-time estimates with confidence intervals that hold up across job types and regions.

### Business Context

HouseAccount matches homeowners with home service providers. Pricing is one of the highest-leverage surfaces in the marketplace because it gates trust on both sides. Homeowners who trust the price book without shopping the market. Providers who trust the price accept without renegotiation. Both behaviors drive our core marketplace metrics: match rate, time to match, and fulfillment rate. Our current pricing logic is a strong baseline. The opportunity is to layer on an AI model that combines internal job data with public knowledge bases for sharper, more defensible estimates with confidence intervals.

### Key Impact Metrics

- Estimate accuracy (predicted vs final price, MAPE)
- Booking conversion rate (homeowner accepts the estimated price)
- Provider acceptance rate (provider accepts at the estimated price without renegotiation)
- Fulfillment rate (jobs completed without price adjustment)

### Required Programming Languages

Modeling work can happen in any framework. The API layer must integrate with HouseAccount's payload shapes.

- We strongly prefer Rails ([style guide](https://raw.githubusercontent.com/HouseAccountEng/guidelines/refs/heads/main/STYLE.md) provided as a resource), but candidates may use a thin Python wrapper (Flask/FastAPI) or Sinatra if not Rails-fluent.
- The integration test is what matters. Your endpoint accepts our booking input and returns our expected response.

### AI/ML Frameworks

Open. Candidates may use OpenAI, Anthropic, open-source models, traditional ML libraries (scikit-learn, XGBoost), or any combination. The model can be fully ML, fully LLM-driven, or hybrid. We care about the modeling approach and the result, not the specific framework.

### Development Tools

- Ruby on Rails for the API layer when the language is a free choice
- Git and GitHub for version control
- AI coding agents (Claude Code, Cursor, etc.) actively encouraged
- Public HouseAccount Rails style guide provided as a link to feed into agents

### Cloud Platforms

No specific cloud requirement. Candidates may run locally or deploy to any platform that lets us reproduce the result.

### Other Specific Requirements

- Must integrate with HouseAccount's booking flow via the staging endpoint we provide (see Functional Requirements)
- Must accept structured booking input and return a structured price estimate plus a confidence score or range
- Service categories: train on all 18 categories in the dataset. The model should produce estimates for any category, but confidence should drop for categories outside the current production set (see Functional Requirements)
- Full API contract (auth, request/response shape, error handling) is documented in Appendix A

### Dataset

A sanitized historical pricing dataset is provided as input: [houseaccount_pricing_sample (Google Sheets, 1,432 rows)](https://docs.google.com/spreadsheets/d/1B1tZILEPD1i_HSkd5rzgPJJYNPaRyBXzs3K3fA9j9pU/edit?gid=1788027344#gid=1788027344). Download as CSV from the File menu in Sheets. Key details:

- **1,432 rows** of home service jobs across **18 categories**
- **Every row** has an AI-generated estimate range (`estimate_lo`, `estimate_hi`, `original_estimate` midpoint). This is the current pricing model output that your model is trying to beat
- **411 rows** have a `final_price` (what the provider actually charged). This is the supervised signal for model training and the basis for evaluation
- **1,033 unique zip codes** (real ZIPs). Candidates may join external census or demographic data
- **No scope fields** (square footage, fixture count, complexity rating) exist in the schema. Candidates extract scope signals from `job_description` text, which is part of the challenge

**Columns available:**
`job_id`, `service_category`, `service_subtype`, `zip_code`, `booking_month`, `job_description`, `estimate_lo`, `estimate_hi`, `original_estimate`, `final_price`, `deadline`

**Sample rows:**

| service_category | service_subtype | zip_code | booking_month | job_description | estimate_lo | estimate_hi | original_estimate | final_price | deadline |
|---|---|---|---|---|---|---|---|---|---|
| Plumbing | Plumbing | 33484 | 2026-04 | Replace kitchen sink shutoff valve (you supply valve) | 120.00 | 180.00 | 150.00 | *null* | As soon as possible |
| Pest Control | Bed bugs | 33324 | 2026-04 | Bed bug heat treatment, 2BR apartment | 486.99 | 1168.77 | 827.88 | 697.03 | Within 1 week |
| Cleaning | Window washing (exterior) | 75062 | 2026-03 | Exterior window wash, 2-story, 20 windows | 172.26 | 344.51 | 258.38 | 247.08 | Flexible |
| Painting | Interior painting | 33463 | 2026-03 | Bathroom walls painting project | 200.00 | 400.00 | 300.00 | *null* | As soon as possible |

**Data preparation notes:**
- All PII stripped (no names, emails, phones, addresses, internal IDs)
- Job IDs are SHA-256 hashes
- Dates truncated to YYYY-MM
- Descriptions regex-scrubbed for contact info and addresses
- 13 outlier rows over $10K dropped (large remodels that skew the distribution)
- Handyman category sampled down to prevent dataset domination
- Sparse categories augmented to ensure adequate coverage for category-specific modeling

### Functional Requirements (Must-Haves)

1. AI pricing model that takes booking input (job type, scope, location, attributes) and returns an estimated price plus a confidence score or range
2. Integration with the HouseAccount booking flow via the [staging endpoint](https://pro.houseparty.dev/api/reference#bookings-create) we provide
3. Documented modeling approach: data used, assumptions made, training or prompting strategy, and how confidence is calculated
4. Code follows the public HouseAccount Rails style guide
5. Working end-to-end prototype: booking input in, price estimate out, posted to the staging endpoint
6. Confidence calibration: confidence must drop below 0.5 for inputs outside the training distribution. Out-of-distribution means an estimate midpoint above $5,000, a prediction interval wider than 3x the median, or a service category outside our current production set. Do not reject or cap these inputs. Pass them through with low confidence so we can route them appropriately

### Code Quality Expectations

- Adheres to the public HouseAccount Rails style guide
- README a new engineer can use to clone, run, and understand the project in under 15 minutes
- Tests for core pricing logic and integration boundaries
- Secrets handled via environment variables (no keys in code or logs)
- Clear separation between data layer, model layer, and API layer

### Performance Benchmarks

- Pricing estimate response time under 2 seconds end to end
- Model accuracy is evaluated on two MAPE numbers computed from your submitted predictions. Blended MAPE on the full 411-row priced subset (baseline 11.6 percent, median APE 8.3 percent) and real-only MAPE on a held-out portion of those rows (baseline ~40 percent). **You must beat baseline on both subsets.** Top finalists may be evaluated on an additional held-out post-snapshot real dataset not shared during the project
- Should handle a typical booking surge volume (load spec to be confirmed at kickoff)

### Time Constraints

Minimum 3 full days of focused work, suitable for weeks 3-4 of the Gauntlet program. No hard upper bound. Candidates should ship the most polished version they can in the time they have.

---

## Step 3: Additional Considerations

### Technical Contact

Claudio is the primary technical contact for questions about Rails conventions, integration, pricing data, and the API schema.

### Off-Limits Technologies or Approaches

- Do not hardcode pricing tables. The point of the project is the model, not a lookup
- Do not store secrets, keys, or credentials in code or logs
- Do not skip the booking system integration. A model in a notebook with no integration will not be evaluated
- Do not scrape competitor pricing in ways that violate terms of service

### AI Usage Documentation

**Required.** Candidates must document their AI tool usage. We are hiring AI Engineers, so how you use AI is part of the signal.

Submit a markdown file `AI_USAGE.md` with these sections:

- **Tools used.** Which AI tools you used and the role each played (coding agent, model selection, prompt iteration, etc.)
- **Significant prompts.** The 5 to 10 prompts that shaped the architecture, not every line of generated code
- **Validation steps for AI-generated code.** Tests run, manual review notes, hallucinations caught
- **Reflection.** Where AI helped most. Where it produced bad output. What you'd do differently next time

A concrete template means comparable submissions and a clearer hiring signal.

### Submission Requirements

- Source Code (GitHub repo)
- Technical Documentation (README plus modeling approach doc)
- Demo Video (short walkthrough of the working prototype)
- Deployment Guide (how to run locally and against the staging endpoint)
- AI Usage Log (`AI_USAGE.md` per the template above)

---

# Appendix A: Pricing Model API Schema

External-facing schema for the HouseAccount AI Pricing Model capstone. Candidates implement a service that accepts a structured booking payload and returns a price estimate with a confidence value. This document is the contract.

Every field, validation rule, and example in this appendix is either traceable to a specific `file:line` in the HouseAccount codebase (Sources section at the end) or explicitly flagged **"new for Gauntlet, no codebase precedent"** with rationale.

## Endpoint

```
POST /.netlify/functions/pricing-estimate
```

Path convention follows the rest of the concierge function surface: kebab-case, singular noun, domain-prefixed with `pricing-` (matches 7 existing `pricing-*` functions in `concierge-dashboard/netlify/functions/`). No `/api/` prefix. Full URL is `/.netlify/functions/pricing-estimate`.

Content type: `application/json` on both request and response. UTF-8.

## Authentication

`Authorization: Bearer <SECRET>` header. Validated with constant-time comparison (`timingSafeEqual`) against an env-var-held shared secret. Missing/wrong header returns `401 { "error": "Unauthorized" }`.

Pattern mirrors `concierge-dashboard/netlify/functions/receive-homeowner.js:75-81` (the newer of HouseAccount's two partner-ingestion endpoints, with rate limiting and audit logging; preferred over the older `receive-booking.js`).

```js
// Verbatim shape candidates should replicate
import { timingSafeEqual } from 'node:crypto'

function validateBearer(req) {
  const authz = req.headers.get('authorization') || ''
  if (!authz.startsWith('Bearer ')) return false
  const presented = Buffer.from(authz.slice('Bearer '.length))
  const expected = Buffer.from(GAUNTLET_PRICING_SECRET || '')
  return presented.length === expected.length && timingSafeEqual(presented, expected)
}
```

**Env var name:** `GAUNTLET_PRICING_SECRET` *(new for Gauntlet, no codebase precedent. Naming follows the established `<PARTNER>_INBOUND_SECRET` pattern used by `FOUNTAIN_INBOUND_SECRET`, `INTAKE_INBOUND_SECRET`, `MOVE_CONCIERGE_INBOUND_SECRET`).*

Boot-time enforcement: the function should throw on cold start if the env var is unset, matching `receive-homeowner.js:20-22`.

## Request

### Required fields

| Field | Type | Notes | Source |
|---|---|---|---|
| `job_id` | string | Stable identifier for this estimate request. Used for idempotency and eval attribution. | New for Gauntlet (mirrors `journey_id` role from `receive-booking.js:79`). |
| `service_category` | string | Free-text category. Train on all 18 categories present in the dataset. The endpoint should accept any string, not strictly validate. See Service categories below. | `start/db/seeds/suggestions.csv` and training dataset |
| `zip_code` | string | 5-digit US ZIP. Real ZIPs in seed data. Candidates may join external census data. | `concierge-dashboard/netlify/functions/receive-homeowner.js:40` (regex `\b(\d{5})(?:-\d{4})?\b`) |
| `job_description` | string | Homeowner free-text describing the job. The scope-extraction surface. No separate fields for square footage, fixture count, etc. | Project brief, Dataset section |

### Optional fields

| Field | Type | Notes | Source |
|---|---|---|---|
| `service_subtype` | string | Sub-classification within `service_category` (e.g. "Tv Mounting" under handyman). Free text. Candidates should treat it as a hint, not a strict validator. | Project brief, Dataset section |
| `deadline` | string | One of: `"As soon as possible"`, `"Within 1-2 weeks"`, `"Within 1 month"`, `"I'm flexible"`. Exact strings, including the straight ASCII apostrophe in `I'm flexible`. | `start/app/controllers/deadlines_controller.rb:7` |
| `booking_month` | string | `YYYY-MM`. Dates in the training dataset are truncated to this resolution. | Project brief, Dataset section |
| `job_status` | string | Free-text status from the homeowner's perspective. Not present in the current training dataset, but the endpoint should accept it if provided. | Production booking flow |
| `original_estimate` | number \| null | Previous-model midpoint, when available. Useful for candidates who want to blend against the baseline. | Project brief, Dataset section |
| `original_estimate_lo` | number \| null | Previous-model low bound. | Project brief, Dataset section |
| `original_estimate_hi` | number \| null | Previous-model high bound. | Project brief, Dataset section |

### Validation rules

- Missing `job_id`, `service_category`, `zip_code`, or `job_description` returns `400 { "error": "<field> required" }`. Validation-message convention is `'<snake_case_field> required'` per `receive-homeowner.js:202-216` and `receive-booking.js:82`.
- Malformed JSON body returns `400 { "error": "Malformed JSON" }` (`receive-booking.js:74`, `receive-homeowner.js:113`).
- Wrong method returns `405 { "error": "Method not allowed" }` JSON body (matches `receive-homeowner.js:87`. Do **not** replicate the inconsistent plain-text 405 in `receive-booking.js:67`).

### Service categories

HouseAccount currently sells 10 production verticals and is expanding into adjacent categories. The training dataset reflects this expansion plan and includes 18 categories:

```
Appliance Repair, Auto, Chimney, Cleaning, Electrical, Exterior,
Flooring, General Contractor, Handyman, HVAC, Landscaping, Moving,
Painting, Pest Control, Plumbing, Pool, Remodeling, Roofing
```

**Current production verticals (10):**

```
electrical, exterior-cleaning, handyman, hvac, indoor-cleaning,
irrigation, landscaping-lawn, pest-control, plumbing, tick-mosquito-treatment
```

**Candidate posture:**

- Train on all 18 categories from the dataset
- Accept any category string as input. Do not strictly validate
- For categories outside the current production set, the model should still produce an estimate, but confidence should reflect reduced certainty
- Surface the category-coverage tradeoff in your model card

Case handling: the training dataset uses title-case ("Plumbing"). The production seed file uses kebab-case ("plumbing"). Candidates should normalize on read.

### Example request

```json
{
  "job_id": "abc123",
  "service_category": "Plumbing",
  "service_subtype": "Water Heater Replacement",
  "zip_code": "78704",
  "job_description": "50-gallon gas water heater stopped working last night, pilot light won't stay lit. Need replacement.",
  "deadline": "Within 1-2 weeks",
  "booking_month": "2026-05",
  "original_estimate": 1850,
  "original_estimate_lo": 1400,
  "original_estimate_hi": 2300
}
```

## Response

### 200 - Success

```json
{
  "ok": true,
  "job_id": "abc123",
  "estimate_lo": 1450,
  "estimate_hi": 2200,
  "estimate_midpoint": 1825,
  "confidence": 0.78,
  "model_version": "candidate-name-v1.2.0"
}
```

Wrapper shape (`{ ok: true, ... }`) matches `receive-booking.js:116,151` and `receive-homeowner.js:164,199`. The endpoint echoes `job_id` instead of `external_id` because pricing is request/response, not idempotent ingestion (no concept of a partner-side primary key).

| Field | Type | Range | Source / Rationale |
|---|---|---|---|
| `ok` | bool | `true` on success | `receive-booking.js:151`, `receive-homeowner.js:199` |
| `job_id` | string | echoes the request | Echoing the inbound ID is standard in partner endpoints (see `external_id` echo in both `receive-*` files). |
| `estimate_lo` | number | low bound of the price range, USD | `start/app/models/assessment.rb:21,25,51,58-60`; `start/app/models/booking/concierge.rb:8`; `start/app/controllers/goldrush/assessments_controller.rb:24`. |
| `estimate_hi` | number | high bound of the price range, USD | Same as above. |
| `estimate_midpoint` | number | point estimate, USD | **New for Gauntlet, no codebase precedent.** Present so HouseAccount can compute MAPE against a single comparable value across candidates regardless of internal modeling approach. Computing `(lo + hi) / 2` server-side assumes a uniform distribution and produces worse MAPE comparisons across candidates with skewed confidence, so we ask candidates to provide it directly. |
| `confidence` | number | 0.0 to 1.0 (inclusive), clamped | `concierge-dashboard/netlify/functions/_lib/classifyRequestClass.js:30,80-83`; `classify-lead-tags.js:53-54`; `classify-end-of-lead-behavior.js:49-50`. HouseAccount convention is a single field named `confidence`, value in `[0, 1]`. Existing threshold convention: `>=0.8` obvious, `0.5â€“0.8` ambiguous, `<0.5` guess (`classifyRequestClass.js:30`). Auto-action threshold elsewhere in the codebase is `>=0.75` (`classify-lead-tags.js:97`). |
| `model_version` | string | candidate-defined, free-form | **New for Gauntlet, no codebase precedent.** Used to attribute predictions to candidate submissions during eval. Recommended format: `<candidate-id>-v<semver>`. Any non-empty string is accepted. |

### Confidence calibration requirements

Confidence must drop below 0.5 for inputs outside the training distribution. Out-of-distribution means:

- `estimate_midpoint` above $5,000 (95th percentile of training data)
- Prediction interval (`estimate_hi - estimate_lo`) wider than 3x the median observed range
- `service_category` outside the current production set of 10 verticals

Do not reject or cap out-of-distribution inputs. Pass them through with low confidence.

### Error responses

All error bodies use the single-key shape `{ "error": "<message>" }` (established convention across both partner-ingest endpoints).

| Code | Body | When | Source |
|---|---|---|---|
| `400` | `{ "error": "Malformed JSON" }` | Body isn't valid JSON | `receive-booking.js:74`, `receive-homeowner.js:113` |
| `400` | `{ "error": "<field> required" }` | A required field is missing/blank | `receive-booking.js:82`, `receive-homeowner.js:202-216` |
| `401` | `{ "error": "Unauthorized" }` | Missing/invalid `Authorization` header | `receive-booking.js:41`, `receive-homeowner.js:93` |
| `405` | `{ "error": "Method not allowed" }` | Non-POST method | `receive-homeowner.js:87` (preferred shape; not `receive-booking.js:67`'s plain-text variant) |
| `429` | `{ "error": "Rate limit exceeded", "retry_after": 60 }` + `Retry-After: 60` header | Optional. If candidates implement rate limiting, match this shape. | `receive-homeowner.js:100-104, 219-234` |
| `500` | `{ "error": "<short reason>" }` | Internal failure. Examples in codebase: `"Upsert failed"`, `"Resubmission record failed"`. Keep messages short and free of stack traces / PII. | `receive-booking.js:114,148`; `receive-homeowner.js:160,195` |

CORS: not required for the Gauntlet endpoint (candidates aren't expected to serve a browser frontend). If a candidate adds CORS, follow the `receive-booking.js:27-31` pattern.

## Outlier handling

13 rows over $10K were dropped from the training set as outliers. The endpoint should not reject or clamp out-of-range inputs. Pass them through with low confidence (see Confidence calibration requirements above). This preserves real jobs that happen to be large while signaling uncertainty downstream.

## Existing booking flow reference

- HouseAccount booking API reference: [start.houseaccount.com/api/reference](https://start.houseaccount.com/api/reference)
- Public HouseAccount Rails style guide: [github.com/HouseAccountEng/guidelines/blob/main/STYLE.md](https://raw.githubusercontent.com/HouseAccountEng/guidelines/refs/heads/main/STYLE.md)

The closest internal analogue for partner ingestion is `concierge-dashboard/netlify/functions/receive-homeowner.js`, recommended as the template for candidates studying HouseAccount conventions.

## Out of scope

- **Load testing.** Performance benchmark load profile is intentionally not specified for this capstone. The 2-second response-time target is per-request. Sustained throughput and concurrency limits are not part of the eval.
- **Caching layer.** Candidates may cache internally but are not required to expose cache controls (`ETag`, `Cache-Control`, etc.) on the response.
- **Webhook callbacks.** Pricing is synchronous request/response. No async webhook pattern like the Goldrush assessment callback (`start/app/models/assessment.rb:42-44`).
- **Multi-tenant isolation.** Single shared bearer secret. No per-candidate scoping at the auth layer.

## Sources

Every verified item in this document, with its `file:line` citation. Items flagged "new for Gauntlet" are listed separately at the bottom.

### Verified from codebase

| Item | Source |
|---|---|
| Bearer auth pattern (validateBearer + timingSafeEqual) | `concierge-dashboard/netlify/functions/receive-homeowner.js:75-81` |
| Bearer auth boot-time enforcement (throw if env var unset) | `concierge-dashboard/netlify/functions/receive-homeowner.js:20-22` |
| Older auth variant (multi-secret) | `concierge-dashboard/netlify/functions/receive-booking.js:16-21, 56-63` |
| Env var naming pattern `<PARTNER>_INBOUND_SECRET` | `receive-booking.js:16-17`, `receive-homeowner.js:18` |
| Deadline canonical strings | `start/app/controllers/deadlines_controller.rb:7` |
| Service categories - 10 vertical slugs | `start/db/seeds/suggestions.csv` (header line 1, data lines 2â€“91); seeded via `start/db/seeds.rb:25-34` |
| ZIP-code regex | `concierge-dashboard/netlify/functions/receive-homeowner.js:40` |
| `estimate_lo` / `estimate_hi` naming | `start/app/models/assessment.rb:21,25,51,58-60`; `start/app/models/booking/concierge.rb:8`; `start/app/controllers/goldrush/assessments_controller.rb:24`; `concierge-dashboard/netlify/functions/receive-booking.js:77` (commented but recognized) |
| `confidence` field (0-1 range, clamped) | `concierge-dashboard/netlify/functions/_lib/classifyRequestClass.js:22,30,35,80-83` |
| `confidence` threshold conventions | `_lib/classifyRequestClass.js:30`; `classify-lead-tags.js:18,97-102`; `classify-end-of-lead-behavior.js:19,113` |
| Response wrapper `{ ok: true, ... }` | `receive-booking.js:116,151`; `receive-homeowner.js:164,199` |
| Echoed-ID-on-success convention | `receive-booking.js:118,151`; `receive-homeowner.js:164,199` |
| Error body shape `{ error: '<msg>' }` | `receive-booking.js:41,74,82,114,148`; `receive-homeowner.js:87,93,113,120,160,195` |
| 401 `Unauthorized` body | `receive-booking.js:41`; `receive-homeowner.js:93` |
| 400 `Malformed JSON` body | `receive-booking.js:74`; `receive-homeowner.js:113` |
| 400 `<field> required` validation messages | `receive-booking.js:82`; `receive-homeowner.js:202-216` |
| 405 JSON body (preferred over plain-text 405) | `receive-homeowner.js:87` (preferred) vs `receive-booking.js:67` (inconsistent, not the pattern) |
| 429 rate limit body + `Retry-After` header | `receive-homeowner.js:100-104, 219-234` |
| 500 short-message body | `receive-booking.js:114,148`; `receive-homeowner.js:160,195` |
| CORS headers shape | `receive-booking.js:27-31` |
| Endpoint naming convention (kebab-case, singular, domain-prefixed) | Survey of `concierge-dashboard/netlify/functions/*.js` (83 files); pricing-* prefix: `pricing-auto-close-runner`, `pricing-draft-reply`, `pricing-eligible-providers`, `pricing-render-template`, `pricing-reply`, `pricing-save`, `pricing-send-sms` |
| No `/api/` prefix on netlify functions | `concierge-dashboard/netlify/functions/` directory layout; URL pattern `/.netlify/functions/<name>` |
| Outbound booking payload from start to concierge | `start/app/models/booking/concierge.rb:5-12` |
| `start` has no public POST /api/bookings (legacy redirects to Fountainhead) | `start/config/routes.rb:25-37` |
| Goldrush async callback pattern (out-of-scope reference) | `start/app/models/assessment.rb:42-44` |

### New for Gauntlet (no codebase precedent)

| Item | Rationale |
|---|---|
| `job_id` as request field | Mirrors `journey_id` role from partner ingestion but with a generic name appropriate for a non-HouseAccount-internal-ID source. |
| `estimate_midpoint` response field | Present so HouseAccount can compute MAPE against a single comparable value across candidates regardless of internal modeling approach. |
| `model_version` response field | Attributes predictions to candidate submissions during eval. |
| `GAUNTLET_PRICING_SECRET` env var name | Follows the established `<PARTNER>_INBOUND_SECRET` naming pattern. |
| Endpoint name `pricing-estimate` | New endpoint, but follows existing `pricing-*` prefix convention. |
| Confidence calibration thresholds for OOD inputs | Required to give HouseAccount a signal we can route on at the marketplace layer. |