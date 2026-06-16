# HouseAccount Pricing API

Rails 8 API proxy that sits in front of the FastAPI ML service. Handles auth, input validation, CORS, and the HouseAccount staging integration.

**Full setup and local development:** see the root [README.md](../README.md).

## Routes

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/estimate` | None | Public homeowner endpoint — generates a `job_id` and proxies to the ML service |
| POST | `/api/book` | None | Submits a booking to HouseAccount staging (`pro.houseparty.dev`) |
| POST | `/.netlify/functions/pricing-estimate` | Bearer | Internal API-to-API estimate endpoint |
| POST | `/.netlify/functions/pricing-estimate-batch` | Bearer | Batch estimates (up to 50) |
| POST | `/.netlify/functions/pricing-outcome` | Bearer | Records a final price outcome and triggers retraining |

## Services

- `PricingServiceClient` — authenticated HTTP proxy to the FastAPI ML service
- `HouseAccountBookingService` — HMAC-signed POST to `pro.houseparty.dev/api/bookings`

## Tests

```
bundle exec rails test
```
