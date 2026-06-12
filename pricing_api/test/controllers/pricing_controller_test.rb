require "test_helper"

class PricingControllerTest < ActionDispatch::IntegrationTest
  # ── Auth ────────────────────────────────────────────────────────────────

  test "returns 401 when Authorization header is missing" do
    post "/.netlify/functions/pricing-estimate",
      params: pricing_payload.to_json,
      headers: { "Content-Type" => "application/json" }

    assert_response 401
    assert_equal "Unauthorized", response.parsed_body["error"]
  end

  test "returns 401 when bearer token is wrong" do
    post "/.netlify/functions/pricing-estimate",
      params: pricing_payload.to_json,
      headers: { "Content-Type" => "application/json",
                 "Authorization" => "Bearer wrong-token" }

    assert_response 401
    assert_equal "Unauthorized", response.parsed_body["error"]
  end

  # ── Input validation ────────────────────────────────────────────────────

  test "returns 400 when body is not valid JSON" do
    post "/.netlify/functions/pricing-estimate",
      params: "not json {{",
      headers: auth_header.merge("Content-Type" => "application/json")

    assert_response 400
    assert_equal "Malformed JSON", response.parsed_body["error"]
  end

  %w[job_id service_category zip_code job_description].each do |field|
    test "returns 400 when #{field} is missing" do
      stub_pricing_service
      post "/.netlify/functions/pricing-estimate",
        params: pricing_payload(field => nil).to_json,
        headers: auth_header.merge("Content-Type" => "application/json")

      assert_response 400
      assert_equal "#{field} required", response.parsed_body["error"]
    end
  end

  # ── Success ─────────────────────────────────────────────────────────────

  test "returns 200 with pricing response on valid request" do
    stub_pricing_service
    post "/.netlify/functions/pricing-estimate",
      params: pricing_payload.to_json,
      headers: auth_header.merge("Content-Type" => "application/json")

    assert_response 200
    body = response.parsed_body
    assert body["ok"]
    assert_equal "test-job-001",   body["job_id"]
    assert_equal 150.0,            body["estimate_lo"]
    assert_equal 250.0,            body["estimate_hi"]
    assert_equal 200.0,            body["estimate_midpoint"]
    assert_in_delta 0.72,          body["confidence"], 0.01
    assert_equal "heila-v1.0.0",   body["model_version"]
  end

  # ── Downstream error handling ───────────────────────────────────────────

  test "returns 500 when pricing service is unreachable" do
    stub_request(:post, "http://localhost:8000/predict").to_raise(Errno::ECONNREFUSED)
    post "/.netlify/functions/pricing-estimate",
      params: pricing_payload.to_json,
      headers: auth_header.merge("Content-Type" => "application/json")

    assert_response 500
    assert_equal "Pricing service unavailable", response.parsed_body["error"]
  end

  test "returns 500 when pricing service times out" do
    stub_request(:post, "http://localhost:8000/predict").to_raise(Net::ReadTimeout)
    post "/.netlify/functions/pricing-estimate",
      params: pricing_payload.to_json,
      headers: auth_header.merge("Content-Type" => "application/json")

    assert_response 500
    assert_equal "Pricing service unavailable", response.parsed_body["error"]
  end

  # ── Method guard ────────────────────────────────────────────────────────

  test "returns 405 for GET requests" do
    get "/.netlify/functions/pricing-estimate",
      headers: auth_header

    assert_response 405
    assert_equal "Method not allowed", response.parsed_body["error"]
  end

  # ── Confidence calibration passthrough ─────────────────────────────────

  test "passes OOD low confidence through from pricing service" do
    stub_pricing_service(body: {
      ok: true, job_id: "test-job-001",
      estimate_lo: 2000.0, estimate_hi: 8000.0,
      estimate_midpoint: 5500.0, confidence: 0.4,
      model_version: "heila-v1.0.0"
    })

    post "/.netlify/functions/pricing-estimate",
      params: pricing_payload(service_category: "Remodeling").to_json,
      headers: auth_header.merge("Content-Type" => "application/json")

    assert_response 200
    assert response.parsed_body["confidence"] <= 0.5,
      "Expected OOD confidence < 0.5, got #{response.parsed_body['confidence']}"
  end
end