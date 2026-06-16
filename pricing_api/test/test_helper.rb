ENV["RAILS_ENV"] ||= "test"
ENV["GAUNTLET_PRICING_SECRET"] = "test-secret-key"

require_relative "../config/environment"
require "rails/test_help"
require "webmock/minitest"

WebMock.disable_net_connect!

module ActiveSupport
  class TestCase
    def pricing_payload(overrides = {})
      {
        job_id:            "test-job-001",
        service_category:  "Plumbing",
        zip_code:          "78704",
        job_description:   "Replace kitchen faucet, homeowner supplies faucet",
        deadline:          "Within 1-2 weeks",
        original_estimate: 200.0
      }.merge(overrides).stringify_keys
    end

    def stub_pricing_service(status: 200, body: nil)
      body ||= {
        ok: true, job_id: "test-job-001",
        estimate_lo: 150.0, estimate_hi: 250.0,
        estimate_midpoint: 200.0, confidence: 0.72,
        model_version: "heila-v1.0.0"
      }
      stub_request(:post, "http://localhost:8001/.netlify/functions/pricing-estimate")
        .to_return(status: status, body: body.to_json, headers: { "Content-Type" => "application/json" })
    end

    def booking_payload(overrides = {})
      {
        name:              "Jane Smith",
        phone:             "555-555-1234",
        zip_code:          "33484",
        job_description:   "Install 3 supplied exterior shutters — labor only.",
        estimate_lo:       150.0,
        estimate_hi:       400.0,
        estimate_midpoint: 275.0,
        model_version:     "heila-v1.0.1",
        deadline:          "Within 1-2 weeks",
      }.merge(overrides).stringify_keys
    end

    def auth_header
      { "Authorization" => "Bearer test-secret-key" }
    end
  end
end