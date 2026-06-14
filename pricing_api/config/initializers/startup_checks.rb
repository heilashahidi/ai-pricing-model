# Fail fast if required env vars are missing
raise "GAUNTLET_PRICING_SECRET must be set" if ENV["GAUNTLET_PRICING_SECRET"].blank?

# Warm up the FastAPI pricing service so the first real request doesn't pay cold-start cost
Rails.application.config.after_initialize do
  Thread.new do
    sleep 1  # let Rails finish booting
    begin
      pricing_url = ENV.fetch("PRICING_SERVICE_URL", "http://localhost:8001/predict")
      uri = URI(pricing_url.sub("/predict", "/health"))
      Net::HTTP.start(uri.host, uri.port, open_timeout: 5, read_timeout: 5) { |h| h.get(uri.path) }
      Rails.logger.info("Pricing service warm-up: OK")
    rescue => e
      Rails.logger.warn("Pricing service warm-up failed (will retry on first request): #{e.message}")
    end
  end
end