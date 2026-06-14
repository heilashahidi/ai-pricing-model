# Deployment Guide — Railway

Two services deploy from this single repository.

| Service | Source | Runtime |
|---------|--------|---------|
| `pricing-ml` | repo root | Python 3.11, FastAPI |
| `pricing-api` | `pricing_api/` subdirectory | Ruby 3.3, Rails 8.1 |

---

## Prerequisites

- [Railway account](https://railway.app)
- Repository pushed to GitHub
- Anthropic API key

---

## Step 1 — Create the project

1. Go to [railway.app](https://railway.app) → **New Project**
2. Select **Empty Project**
3. Name it `houseaccount-pricing`

---

## Step 2 — Deploy the FastAPI model service

1. In the project, click **New** → **GitHub Repo** → select this repo
2. Railway creates a service. Rename it **pricing-ml**
3. In Settings → Build:
   - **Root Directory**: `/` (default — leave blank)
   - **Config File Path**: `/railway.json`
4. In Settings → Environment, add variables:

| Variable | Value |
|----------|-------|
| `ANTHROPIC_API_KEY` | `sk-ant-...` |

5. Click **Deploy**. Build takes ~3 minutes (installs Python deps, warms pgeocode cache).
6. After deploy: confirm health at `https://pricing-ml.up.railway.app/health`

---

## Step 3 — Deploy the Rails API + UI

1. In the same project, click **New** → **GitHub Repo** → select this repo again
2. Rename the service **pricing-api**
3. In Settings → Build:
   - **Root Directory**: `/pricing_api`
   - **Config File Path**: `/pricing_api/railway.json`
4. In Settings → Environment, add variables:

| Variable | Value |
|----------|-------|
| `GAUNTLET_PRICING_SECRET` | Choose a strong secret string |
| `RAILS_MASTER_KEY` | Contents of `pricing_api/config/master.key` (run `cat pricing_api/config/master.key` locally) |
| `RAILS_ENV` | `production` |
| `PRICING_SERVICE_URL` | `http://pricing-ml.railway.internal:8000/.netlify/functions/pricing-estimate` |
| `HA_SIGNING_SECRET` | `<your-HA_SIGNING_SECRET>` |
| `HA_APP_NAME` | `gauntlet` |

5. Click **Deploy**

---

## Step 4 — Verify end-to-end

Once both services are green:

```bash
# Replace with your actual Railway domain and secret
curl -X POST https://pricing-api.up.railway.app/.netlify/functions/pricing-estimate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_GAUNTLET_PRICING_SECRET" \
  -d '{
    "job_id": "deploy-test-001",
    "service_category": "Handyman",
    "zip_code": "33484",
    "job_description": "Install 3 supplied exterior shutters (we supply all hardware)",
    "deadline": "Within 1-2 weeks",
    "original_estimate": 750
  }'
```

Expected response:
```json
{
  "ok": true,
  "job_id": "deploy-test-001",
  "estimate_lo": 301.36,
  "estimate_hi": 517.81,
  "estimate_midpoint": 427.25,
  "confidence": 0.66,
  "model_version": "heila-v1.0.0"
}
```

Visit `https://pricing-api.up.railway.app` in a browser to open the demo UI.

---

## Inter-service networking

The two services communicate over Railway's private network:

```
pricing-api (Rails) → http://pricing-ml.railway.internal:8000/.netlify/functions/pricing-estimate
```

Railway's Wireguard-based private network is zero-configuration — no VPC or firewall rules needed. Internal traffic uses `http://`, not `https://`.

---

## Environment variables reference

### pricing-ml (FastAPI)

| Variable | Required | Notes |
|----------|----------|-------|
| `ANTHROPIC_API_KEY` | Yes | Claude Haiku for scope extraction |
| `GAUNTLET_PRICING_SECRET` | Yes | Shared with pricing-api; used for Bearer auth |
| `PORT` | Auto-injected | Railway sets this; FastAPI binds to it |

### pricing-api (Rails)

| Variable | Required | Notes |
|----------|----------|-------|
| `GAUNTLET_PRICING_SECRET` | Yes | Bearer token for both inbound API auth and outbound ML calls |
| `RAILS_MASTER_KEY` | Yes | From `config/master.key` — never commit this |
| `RAILS_ENV` | Yes | Set to `production` |
| `PRICING_SERVICE_URL` | Yes | `http://pricing-ml.railway.internal:8000/.netlify/functions/pricing-estimate` |
| `HA_SIGNING_SECRET` | Yes | HouseAccount staging HMAC key |
| `HA_APP_NAME` | Yes | `gauntlet` |
| `PORT` | Auto-injected | Thruster reads this automatically |
| `RAILS_SERVE_STATIC_FILES` | Auto | Set by Railway; enables UI serving |

---

## Redeployment after model retraining

The model `.joblib` files are baked into the `pricing-ml` Docker image. After running `python3 train.py` locally:

```bash
git add models/
git commit -m "retrain: update model artifacts"
git push
```

Railway auto-deploys on push. The new image will contain the updated models.

---

## Local development

See [README.md](README.md) for local setup instructions.
