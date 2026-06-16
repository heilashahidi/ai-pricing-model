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
    stub_request(:post, "http://localhost:8001/.netlify/functions/pricing-estimate").to_raise(Errno::ECONNREFUSED)
    post "/.netlify/functions/pricing-estimate",
      params: pricing_payload.to_json,
      headers: auth_header.merge("Content-Type" => "application/json")

    assert_response 500
    assert_equal "Pricing service unavailable", response.parsed_body["error"]
  end

  test "returns 500 when pricing service times out" do
    stub_request(:post, "http://localhost:8001/.netlify/functions/pricing-estimate").to_raise(Net::ReadTimeout)
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

  # ── Public homeowner endpoint ───────────────────────────────────────────

  test "public endpoint returns 200 without auth header" do
    stub_pricing_service
    post "/api/estimate",
      params: { service_category: "Handyman", zip_code: "33484",
                job_description:  "Install 3 supplied exterior shutters" }.to_json,
      headers: { "Content-Type" => "application/json" }

    assert_response 200
    assert response.parsed_body["ok"]
  end

  test "public endpoint generates job_id server-side" do
    stub_pricing_service
    post "/api/estimate",
      params: { service_category: "Handyman", zip_code: "33484",
                job_description:  "Install shutters" }.to_json,
      headers: { "Content-Type" => "application/json" }

    assert_response 200
    # job_id is echoed back from the pricing service stub — confirm it was set
    assert response.parsed_body["job_id"].present?
  end

  test "public endpoint returns 400 when service_category missing" do
    post "/api/estimate",
      params: { zip_code: "33484", job_description: "Fix sink" }.to_json,
      headers: { "Content-Type" => "application/json" }

    assert_response 400
    assert_equal "service_category required", response.parsed_body["error"]
  end

  test "public endpoint returns 400 for malformed JSON" do
    post "/api/estimate",
      params: "not json {{",
      headers: { "Content-Type" => "application/json" }

    assert_response 400
    assert_equal "Malformed JSON", response.parsed_body["error"]
  end

  # ── Book action ────────────────────────────────────────────────────────

  test "book returns 200 when HouseAccount accepts the booking" do
    stub_request(:post, "https://pro.houseparty.dev/api/bookings")
      .to_return(status: 201, body: { ok: true, booking_id: "b-123" }.to_json,
                 headers: { "Content-Type" => "application/json" })

    post "/api/book",
      params: booking_payload.to_json,
      headers: { "Content-Type" => "application/json" }

    assert_response 201
    assert response.parsed_body["ok"]
  end

  test "book returns 400 for malformed JSON" do
    post "/api/book",
      params: "not json {{",
      headers: { "Content-Type" => "application/json" }

    assert_response 400
    assert_equal "Malformed JSON", response.parsed_body["error"]
  end

  %w[name phone zip_code job_description estimate_lo estimate_hi].each do |field|
    test "book returns 400 when #{field} is missing" do
      post "/api/book",
        params: booking_payload(field => nil).to_json,
        headers: { "Content-Type" => "application/json" }

      assert_response 400
      assert_equal "#{field} required", response.parsed_body["error"]
    end
  end

  test "book returns 502 when HouseAccount is unreachable" do
    stub_request(:post, "https://pro.houseparty.dev/api/bookings")
      .to_raise(Errno::ECONNREFUSED)

    post "/api/book",
      params: booking_payload.to_json,
      headers: { "Content-Type" => "application/json" }

    assert_response 502
    assert_equal "Booking service unavailable", response.parsed_body["error"]
  end

  test "book returns 502 on HouseAccount SSL error" do
    stub_request(:post, "https://pro.houseparty.dev/api/bookings")
      .to_raise(OpenSSL::SSL::SSLError)

    post "/api/book",
      params: booking_payload.to_json,
      headers: { "Content-Type" => "application/json" }

    assert_response 502
    assert_equal "Booking service unavailable", response.parsed_body["error"]
  end

  # ── estimate_batch ─────────────────────────────────────────────────────

  test "estimate_batch returns 401 without auth" do
    post "/.netlify/functions/pricing-estimate-batch",
      params: { estimates: [ pricing_payload ] }.to_json,
      headers: { "Content-Type" => "application/json" }

    assert_response 401
  end

  test "estimate_batch returns 405 for GET" do
    get "/.netlify/functions/pricing-estimate-batch",
      headers: auth_header

    assert_response 405
    assert_equal "Method not allowed", response.parsed_body["error"]
  end

  test "estimate_batch returns 400 when estimates is missing" do
    post "/.netlify/functions/pricing-estimate-batch",
      params: {}.to_json,
      headers: auth_header.merge("Content-Type" => "application/json")

    assert_response 400
    assert_equal "estimates required", response.parsed_body["error"]
  end

  test "estimate_batch returns 400 when estimates is not an array" do
    post "/.netlify/functions/pricing-estimate-batch",
      params: { estimates: "not-an-array" }.to_json,
      headers: auth_header.merge("Content-Type" => "application/json")

    assert_response 400
    assert_equal "estimates must be an array", response.parsed_body["error"]
  end

  test "estimate_batch proxies to batch service and returns result" do
    stub_request(:post, "http://localhost:8001/.netlify/functions/pricing-estimate-batch")
      .to_return(status: 200,
                 body: { ok: true, results: [ { ok: true, estimate_midpoint: 200.0 } ] }.to_json,
                 headers: { "Content-Type" => "application/json" })

    post "/.netlify/functions/pricing-estimate-batch",
      params: { estimates: [ pricing_payload ] }.to_json,
      headers: auth_header.merge("Content-Type" => "application/json")

    assert_response 200
    assert response.parsed_body["ok"]
  end

  # ── outcome ────────────────────────────────────────────────────────────

  test "outcome returns 401 without auth" do
    post "/.netlify/functions/pricing-outcome",
      params: { job_id: "abc", final_price: 250.0 }.to_json,
      headers: { "Content-Type" => "application/json" }

    assert_response 401
  end

  test "outcome returns 405 for GET" do
    get "/.netlify/functions/pricing-outcome",
      headers: auth_header

    assert_response 405
    assert_equal "Method not allowed", response.parsed_body["error"]
  end

  test "outcome returns 400 when job_id is missing" do
    post "/.netlify/functions/pricing-outcome",
      params: { final_price: 250.0 }.to_json,
      headers: auth_header.merge("Content-Type" => "application/json")

    assert_response 400
    assert_equal "job_id required", response.parsed_body["error"]
  end

  test "outcome returns 400 when final_price is missing" do
    post "/.netlify/functions/pricing-outcome",
      params: { job_id: "abc" }.to_json,
      headers: auth_header.merge("Content-Type" => "application/json")

    assert_response 400
    assert_equal "final_price required", response.parsed_body["error"]
  end

  test "outcome returns 400 when final_price is not a positive number" do
    post "/.netlify/functions/pricing-outcome",
      params: { job_id: "abc", final_price: -50.0 }.to_json,
      headers: auth_header.merge("Content-Type" => "application/json")

    assert_response 400
    assert_equal "final_price must be a positive number", response.parsed_body["error"]
  end

  test "outcome proxies to outcome service on valid request" do
    stub_request(:post, "http://localhost:8001/.netlify/functions/pricing-outcome")
      .to_return(status: 200,
                 body: { ok: true }.to_json,
                 headers: { "Content-Type" => "application/json" })

    post "/.netlify/functions/pricing-outcome",
      params: { job_id: "abc-123", final_price: 250.0 }.to_json,
      headers: auth_header.merge("Content-Type" => "application/json")

    assert_response 200
    assert response.parsed_body["ok"]
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