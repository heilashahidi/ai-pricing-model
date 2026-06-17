# frozen_string_literal: true

class PricingController < ApplicationController
  REQUIRED_FIELDS         = %w[job_id service_category zip_code job_description].freeze
  PUBLIC_REQUIRED_FIELDS  = %w[service_category zip_code job_description].freeze
  OUTCOME_REQUIRED_FIELDS = %w[job_id final_price].freeze
  BOOK_REQUIRED_FIELDS    = %w[name phone zip_code job_description estimate_lo estimate_hi].freeze
  # All three ML endpoints share one base. Deriving OUTCOME/BATCH from the same
  # base as PRICING_SERVICE_URL means one configured URL is enough — a missing
  # OUTCOME_SERVICE_URL can no longer silently fall back to localhost in prod.
  _ml_base = (ENV["PRICING_SERVICE_URL"] ||
    "http://localhost:8001/.netlify/functions/pricing-estimate").sub(%r{/[^/]+\z}, "")
  PRICING_ML_BASE     = ENV.fetch("PRICING_ML_BASE", _ml_base)
  PRICING_SERVICE_URL = ENV.fetch("PRICING_SERVICE_URL", "#{PRICING_ML_BASE}/pricing-estimate")
  BATCH_SERVICE_URL   = ENV.fetch("BATCH_SERVICE_URL",   "#{PRICING_ML_BASE}/pricing-estimate-batch")
  OUTCOME_SERVICE_URL = ENV.fetch("OUTCOME_SERVICE_URL", "#{PRICING_ML_BASE}/pricing-outcome")

  def public_estimate
    body = parse_body
    return render_error 400, "Malformed JSON" if body.nil?
    missing = PUBLIC_REQUIRED_FIELDS.find { |field| body[field].blank? }
    return render_error 400, "#{missing} required" if missing
    payload = body.slice(*PUBLIC_REQUIRED_FIELDS + %w[deadline service_subtype])
    payload["job_id"]        = SecureRandom.uuid
    payload["booking_month"] = Time.current.strftime("%Y-%m")
    result = PricingServiceClient.new.call PRICING_SERVICE_URL, payload
    render json: result[:body], status: result[:status]
  end

  def book
    body = parse_body
    return render_error 400, "Malformed JSON" if body.nil?
    missing = BOOK_REQUIRED_FIELDS.find { |field| body[field].blank? }
    return render_error 400, "#{missing} required" if missing
    result = HouseAccountBookingService.new.submit body
    render json: result[:body], status: result[:status]
  end

  def estimate_batch
    return render_error 405, "Method not allowed" unless request.post?
    return render_error 401, "Unauthorized" unless authenticate
    body = parse_body
    return render_error 400, "Malformed JSON" if body.nil?
    return render_error 400, "estimates required" if body["estimates"].blank?
    return render_error 400, "estimates must be an array" unless body["estimates"].is_a? Array
    result = PricingServiceClient.new.call BATCH_SERVICE_URL, body
    render json: result[:body], status: result[:status]
  end

  def outcome
    return render_error 405, "Method not allowed" unless request.post?
    return render_error 401, "Unauthorized" unless authenticate
    body = parse_body
    return render_error 400, "Malformed JSON" if body.nil?
    missing = OUTCOME_REQUIRED_FIELDS.find { |field| body[field].blank? }
    return render_error 400, "#{missing} required" if missing
    price = body["final_price"]
    return render_error 400, "final_price must be a positive number" unless price.is_a? Numeric
    return render_error 400, "final_price must be a positive number" unless price.positive?
    payload = body.slice(*OUTCOME_REQUIRED_FIELDS).merge("source" => "verified")
    result = PricingServiceClient.new.call OUTCOME_SERVICE_URL, payload
    render json: result[:body], status: result[:status]
  end

  # Homeowner-facing outcome submission. No Bearer required; recorded as "demo"
  # so untrusted public input is never used for model retraining.
  def public_outcome
    body = parse_body
    return render_error 400, "Malformed JSON" if body.nil?
    missing = OUTCOME_REQUIRED_FIELDS.find { |field| body[field].blank? }
    return render_error 400, "#{missing} required" if missing
    price = body["final_price"]
    return render_error 400, "final_price must be a positive number" unless price.is_a?(Numeric) && price.positive?
    payload = body.slice(*OUTCOME_REQUIRED_FIELDS).merge("source" => "demo")
    result = PricingServiceClient.new.call OUTCOME_SERVICE_URL, payload
    render json: result[:body], status: result[:status]
  end

  def estimate
    return render_error 405, "Method not allowed" unless request.post?
    return render_error 401, "Unauthorized" unless authenticate
    body = parse_body
    return render_error 400, "Malformed JSON" if body.nil?
    missing = REQUIRED_FIELDS.find { |field| body[field].blank? }
    return render_error 400, "#{missing} required" if missing
    result = PricingServiceClient.new.call PRICING_SERVICE_URL, body
    render json: result[:body], status: result[:status]
  end

private

  def authenticate
    header = request.headers["Authorization"] || ""
    return false unless header.start_with? "Bearer "
    presented = header.delete_prefix "Bearer "
    expected  = ENV["GAUNTLET_PRICING_SECRET"].to_s
    return false if expected.empty?
    ActiveSupport::SecurityUtils.secure_compare presented, expected
  end

  def parse_body
    JSON.parse request.body.read
  rescue JSON::ParserError
    nil
  end

  def render_error(status, message)
    render json: { error: message }, status: status
  end
end
