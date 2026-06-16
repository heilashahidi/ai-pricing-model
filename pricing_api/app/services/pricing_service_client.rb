class PricingServiceClient
  TIMEOUT = 8

  def call(url, body)
    uri  = URI(url)
    http = Net::HTTP.new(uri.host, uri.port)
    http.open_timeout = TIMEOUT
    http.read_timeout = TIMEOUT

    req = Net::HTTP::Post.new(uri.path,
      "Content-Type"  => "application/json",
      "Authorization" => "Bearer #{ENV.fetch("GAUNTLET_PRICING_SECRET", "")}")
    req.body = body.to_json

    response = http.request(req)
    { status: response.code.to_i, body: JSON.parse(response.body) }
  rescue Net::OpenTimeout, Net::ReadTimeout, Errno::ECONNREFUSED,
         Errno::ETIMEDOUT, SocketError, EOFError => error
    Rails.logger.error("Pricing service unavailable (#{url}): #{error.class} #{error.message}")
    { status: 500, body: { error: "Pricing service unavailable" } }
  rescue JSON::ParserError => error
    Rails.logger.error("Pricing service bad response (#{url}): #{error.message}")
    { status: 500, body: { error: "Internal pricing error" } }
  end
end
