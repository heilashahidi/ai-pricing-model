require "test_helper"

class PricingServiceClientTest < ActiveSupport::TestCase
  URL = "http://localhost:8001/.netlify/functions/pricing-estimate"

  setup do
    @client  = PricingServiceClient.new
    @payload = { "job_id" => "test-001", "service_category" => "Plumbing",
                 "zip_code" => "33484", "job_description" => "Fix faucet" }
  end

  test "returns parsed body and status on success" do
    response_body = { ok: true, estimate_midpoint: 200.0, confidence: 0.75 }.to_json
    stub_request(:post, URL).to_return(status: 200, body: response_body,
                                       headers: { "Content-Type" => "application/json" })

    result = @client.call(URL, @payload)

    assert_equal 200, result[:status]
    assert_equal 200.0, result[:body]["estimate_midpoint"]
  end

  test "returns 500 on connection refused" do
    stub_request(:post, URL).to_raise(Errno::ECONNREFUSED)

    result = @client.call(URL, @payload)

    assert_equal 500, result[:status]
    assert_equal "Pricing service unavailable", result[:body][:error]
  end

  test "returns 500 on read timeout" do
    stub_request(:post, URL).to_raise(Net::ReadTimeout)

    result = @client.call(URL, @payload)

    assert_equal 500, result[:status]
    assert_equal "Pricing service unavailable", result[:body][:error]
  end

  test "returns 500 on open timeout" do
    stub_request(:post, URL).to_raise(Net::OpenTimeout)

    result = @client.call(URL, @payload)

    assert_equal 500, result[:status]
    assert_equal "Pricing service unavailable", result[:body][:error]
  end

  test "returns 500 when response body is not valid JSON" do
    stub_request(:post, URL).to_return(status: 200, body: "not json {{")

    result = @client.call(URL, @payload)

    assert_equal 500, result[:status]
    assert_equal "Internal pricing error", result[:body][:error]
  end

  test "sends Authorization header with secret" do
    original = ENV["GAUNTLET_PRICING_SECRET"]
    ENV["GAUNTLET_PRICING_SECRET"] = "test-secret"
    stub_request(:post, URL)
      .to_return(status: 200, body: { ok: true }.to_json,
                 headers: { "Content-Type" => "application/json" })

    result = @client.call(URL, @payload)
    assert_equal 200, result[:status]
    assert_requested(:post, URL) { |req| req.headers["Authorization"] == "Bearer test-secret" }
  ensure
    ENV["GAUNTLET_PRICING_SECRET"] = original
  end
end
