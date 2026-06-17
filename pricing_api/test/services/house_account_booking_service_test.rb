require "test_helper"

class HouseAccountBookingServiceTest < ActiveSupport::TestCase
  ENDPOINT = "https://pro.houseparty.dev/api/bookings"

  setup do
    @service = HouseAccountBookingService.new
    @body    = {
      "name"              => "Jane Smith",
      "phone"             => "555-555-1234",
      "zip_code"          => "33484",
      "job_description"   => "Install 3 supplied exterior shutters — labor only.",
      "estimate_lo"       => 150.0,
      "estimate_hi"       => 400.0,
      "estimate_midpoint" => 275.0,
      "model_version"     => "houseaccount-v1.0.1",
      "deadline"          => "Within 1-2 weeks",
    }
  end

  test "returns 201 on successful HouseAccount booking" do
    stub_request(:post, ENDPOINT)
      .to_return(status: 201, body: { ok: true, booking_id: "b-123" }.to_json,
                 headers: { "Content-Type" => "application/json" })

    result = @service.submit(@body)

    assert_equal 201, result[:status]
    assert result[:body]["ok"]
  end

  test "sends correct payload shape to HouseAccount" do
    stub = stub_request(:post, ENDPOINT)
             .with(body: hash_including(
               "name"     => "Jane Smith",
               "zip"      => "33484",
               "phone"    => "555-555-1234",
               "estimate" => { "min" => 150, "max" => 400 }
             ))
             .to_return(status: 201, body: { ok: true }.to_json,
                        headers: { "Content-Type" => "application/json" })

    @service.submit(@body)
    assert_requested stub
  end

  test "truncates job_description to 200 characters in summary" do
    @body["job_description"] = "a" * 300

    stub = stub_request(:post, ENDPOINT)
             .with(body: hash_including("summary" => "#{"a" * 197}..."))
             .to_return(status: 201, body: { ok: true }.to_json,
                        headers: { "Content-Type" => "application/json" })

    @service.submit(@body)
    assert_requested stub
  end

  test "returns 502 on connection refused" do
    stub_request(:post, ENDPOINT).to_raise(Errno::ECONNREFUSED)

    result = @service.submit(@body)

    assert_equal 502, result[:status]
    assert_equal "Booking service unavailable", result[:body][:error]
  end

  test "returns 502 on SSL error" do
    stub_request(:post, ENDPOINT).to_raise(OpenSSL::SSL::SSLError)

    result = @service.submit(@body)

    assert_equal 502, result[:status]
    assert_equal "Booking service unavailable", result[:body][:error]
  end

  test "returns 502 on read timeout" do
    stub_request(:post, ENDPOINT).to_raise(Net::ReadTimeout)

    result = @service.submit(@body)

    assert_equal 502, result[:status]
    assert_equal "Booking service unavailable", result[:body][:error]
  end

  test "returns 502 when HouseAccount returns non-JSON body" do
    stub_request(:post, ENDPOINT).to_return(status: 500, body: "Internal Server Error")

    result = @service.submit(@body)

    assert_equal 502, result[:status]
    assert_equal "Booking service unavailable", result[:body][:error]
  end

  test "sends App-Name App-Timestamp App-Signature headers" do
    stub_request(:post, ENDPOINT)
      .to_return(status: 201, body: { ok: true }.to_json,
                 headers: { "Content-Type" => "application/json" })

    @service.submit(@body)

    assert_requested(:post, ENDPOINT) do |req|
      req.headers["App-Name"] == "gauntlet" &&
        req.headers["App-Timestamp"].match?(/\A\d+\z/) &&
        req.headers["App-Signature"].present?
    end
  end
end
