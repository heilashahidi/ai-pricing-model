ENV["RAILS_ENV"] ||= "test"
ENV["GAUNTLET_PRICING_SECRET"] = "test-secret-key"

require_relative "../config/environment"
require "rails/test_help"
require "webmock/minitest"

WebMock.disable_net_connect!(allow_localhost: true)

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
      stub_request(:post, "http://localhost:8000/predict")
        .to_return(status: status, body: body.to_json, headers: { "Content-Type" => "application/json" })
    end

    def auth_header
      { "Authorization" => "Bearer test-secret-key" }
    end
  end
end