Rails.application.config.middleware.insert_before 0, Rack::Cors do
  allow do
    # Wildcard intentional — /api/estimate and /api/book are public, unauthenticated
    # endpoints designed for browser clients on any domain. The /.netlify/functions/*
    # endpoints are omitted here; they require a Bearer token and are API-to-API only.
    origins "*"

    resource "/api/estimate",
      headers: :any,
      methods: [ :post, :options ]

    resource "/api/book",
      headers: :any,
      methods: [ :post, :options ]
  end
end
