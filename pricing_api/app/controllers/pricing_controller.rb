class PricingController < ApplicationController
  REQUIRED_FIELDS         = %w[job_id service_category zip_code job_description].freeze
  PUBLIC_REQUIRED_FIELDS  = %w[service_category zip_code job_description].freeze
  OUTCOME_REQUIRED_FIELDS = %w[job_id final_price].freeze
  BOOK_REQUIRED_FIELDS    = %w[name phone zip_code job_description estimate_lo estimate_hi].freeze
  PRICING_SERVICE_URL     = ENV.fetch("PRICING_SERVICE_URL",  "http://localhost:8001/.netlify/functions/pricing-estimate")
  BATCH_SERVICE_URL       = ENV.fetch("BATCH_SERVICE_URL",    "http://localhost:8001/.netlify/functions/pricing-estimate-batch")
  OUTCOME_SERVICE_URL     = ENV.fetch("OUTCOME_SERVICE_URL",  "http://localhost:8001/.netlify/functions/pricing-outcome")

  # ── Public homeowner endpoint — no auth required from the browser ──────────
  def public_estimate
    body = parse_body
    return render_error(400, "Malformed JSON") if body.nil?

    missing = PUBLIC_REQUIRED_FIELDS.find { |field| body[field].blank? }
    return render_error(400, "#{missing} required") if missing

    payload = body.slice(*PUBLIC_REQUIRED_FIELDS + %w[deadline service_subtype])
    payload["job_id"]        = SecureRandom.uuid
    payload["booking_month"] = Time.current.strftime("%Y-%m")

    result = PricingServiceClient.new.call(PRICING_SERVICE_URL, payload)
    render json: result[:body], status: result[:status]
  end

  # ── Homeowner booking — submits estimate to HouseAccount staging ───────────
  def book
    body = parse_body
    return render_error(400, "Malformed JSON") if body.nil?

    missing = BOOK_REQUIRED_FIELDS.find { |field| body[field].blank? }
    return render_error(400, "#{missing} required") if missing

    result = HouseAccountBookingService.new.submit(body)
    render json: result[:body], status: result[:status]
  end

  # ── Batch pricing — up to 50 estimates in one round-trip ──────────────────
  def estimate_batch
    return render_error(405, "Method not allowed") unless request.post?

    auth_result = authenticate
    return render_error(401, "Unauthorized") unless auth_result

    body = parse_body
    return render_error(400, "Malformed JSON") if body.nil?
    return render_error(400, "estimates required") if body["estimates"].blank?
    return render_error(400, "estimates must be an array") unless body["estimates"].is_a?(Array)

    result = PricingServiceClient.new.call(BATCH_SERVICE_URL, body)
    render json: result[:body], status: result[:status]
  end

  # ── Outcome callback — called by HouseAccount when a job closes ───────────
  def outcome
    return render_error(405, "Method not allowed") unless request.post?

    auth_result = authenticate
    return render_error(401, "Unauthorized") unless auth_result

    body = parse_body
    return render_error(400, "Malformed JSON") if body.nil?

    missing = OUTCOME_REQUIRED_FIELDS.find { |field| body[field].blank? }
    return render_error(400, "#{missing} required") if missing

    unless body["final_price"].is_a?(Numeric) && body["final_price"].positive?
      return render_error(400, "final_price must be a positive number")
    end

    result = PricingServiceClient.new.call(OUTCOME_SERVICE_URL, body.slice(*OUTCOME_REQUIRED_FIELDS))
    render json: result[:body], status: result[:status]
  end

  # ── Internal API-to-API endpoint — Bearer required ─────────────────────────
  def estimate
    return render_error(405, "Method not allowed") unless request.post?

    auth_result = authenticate
    return render_error(401, "Unauthorized") unless auth_result

    body = parse_body
    return render_error(400, "Malformed JSON") if body.nil?

    missing = REQUIRED_FIELDS.find { |field| body[field].blank? }
    return render_error(400, "#{missing} required") if missing

    result = PricingServiceClient.new.call(PRICING_SERVICE_URL, body)
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

  def render_error(status, message)
    render json: { error: message }, status: status
  end
end
