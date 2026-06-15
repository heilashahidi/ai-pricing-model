class PricingController < ApplicationController
  REQUIRED_FIELDS        = %w[job_id service_category zip_code job_description].freeze
  PUBLIC_REQUIRED_FIELDS = %w[service_category zip_code job_description].freeze
  PRICING_SERVICE_URL    = ENV.fetch("PRICING_SERVICE_URL", "http://localhost:8001/.netlify/functions/pricing-estimate")
  PRICING_SERVICE_TIMEOUT = 8 # seconds — leaves headroom under the 2s SLA

  # ── Public homeowner endpoint — no auth required from the browser ──────────
  def public_estimate
    body = parse_body
    return render_error(400, "Malformed JSON") if body.nil?

    missing = PUBLIC_REQUIRED_FIELDS.find { |f| body[f].blank? }
    return render_error(400, "#{missing} required") if missing

    payload = body.slice(*PUBLIC_REQUIRED_FIELDS + %w[deadline service_subtype])
    payload["job_id"]        = SecureRandom.uuid
    payload["booking_month"] = Time.current.strftime("%Y-%m")

    result = call_pricing_service(payload)
    render json: result[:body], status: result[:status]
  end

  # ── Internal API-to-API endpoint — Bearer required ─────────────────────────
  def estimate
    return render_error(405, "Method not allowed") unless request.post?

    auth_result = authenticate
    return render_error(401, "Unauthorized") unless auth_result

    body = parse_body
    return render_error(400, "Malformed JSON") if body.nil?

    missing = REQUIRED_FIELDS.find { |f| body[f].blank? }
    return render_error(400, "#{missing} required") if missing

    result = call_pricing_service(body)
    render json: result[:body], status: result[:status]
  end

  private

  def authenticate
    header = request.headers["Authorization"] || ""
    return false unless header.start_with?("Bearer ")

    presented = header.delete_prefix("Bearer ")
    expected  = ENV["GAUNTLET_PRICING_SECRET"].to_s
    return false if expected.empty?

    ActiveSupport::SecurityUtils.secure_compare(presented, expected)
  end

  def parse_body
    raw = request.body.read
    JSON.parse(raw)
  rescue JSON::ParserError
    nil
  end

  def call_pricing_service(body)
    uri     = URI(PRICING_SERVICE_URL)
    http    = Net::HTTP.new(uri.host, uri.port)
    http.open_timeout = PRICING_SERVICE_TIMEOUT
    http.read_timeout = PRICING_SERVICE_TIMEOUT

    req           = Net::HTTP::Post.new(uri.path, "Content-Type" => "application/json",
                                                  "Authorization" => "Bearer #{ENV.fetch("GAUNTLET_PRICING_SECRET", "")}")
    req.body      = body.to_json
    response      = http.request(req)
    parsed        = JSON.parse(response.body)

    { status: response.code.to_i, body: parsed }
  rescue Net::OpenTimeout, Net::ReadTimeout, Errno::ECONNREFUSED, Errno::ETIMEDOUT, SocketError, EOFError => e
    Rails.logger.error("Pricing service unavailable: #{e.class} #{e.message}")
    { status: 500, body: { error: "Pricing service unavailable" } }
  rescue JSON::ParserError => e
    Rails.logger.error("Pricing service bad response: #{e.message}")
    { status: 500, body: { error: "Internal pricing error" } }
  end

  def render_error(status, message)
    render json: { error: message }, status: status
  end
end