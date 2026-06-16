Rails.application.config.middleware.insert_before 0, Rack::Cors do
  allow do
    origins "*"

    resource "/api/estimate",
      headers: :any,
      methods: [ :post, :options ]

    resource "/api/book",
      headers: :any,
      methods: [ :post, :options ]
  end
end
