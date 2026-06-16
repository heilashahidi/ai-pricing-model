# frozen_string_literal: true

class HouseAccountBookingService
  ENDPOINT = "https://pro.houseparty.dev/api/bookings"
  TIMEOUT  = 10

  def submit(body)
    payload      = build_payload body
    payload_body = payload.to_json
    timestamp    = Time.current.to_i.to_s

    uri  = URI ENDPOINT
    http = Net::HTTP.new uri.host, uri.port
    http.use_ssl      = true
    http.open_timeout = TIMEOUT
    http.read_timeout = TIMEOUT

    req = Net::HTTP::Post.new uri.path,
      "Content-Type"  => "application/json",
      "App-Name"      => ENV.fetch("HA_APP_NAME", "gauntlet"),
      "App-Timestamp" => timestamp,
      "App-Signature" => sign(timestamp, payload_body)
    req.body = payload_body

    response = http.request req
    { status: response.code.to_i, body: JSON.parse(response.body) }
  rescue Net::OpenTimeout, Net::ReadTimeout, Errno::ECONNREFUSED,
         Errno::ETIMEDOUT, SocketError, EOFError, OpenSSL::SSL::SSLError => error
    Rails.logger.error "HouseAccount booking failed: #{error.class} #{error.message}"
    { status: 502, body: { error: "Booking service unavailable" } }
  rescue JSON::ParserError => error
    Rails.logger.error "HouseAccount bad response: #{error.message}"
    { status: 502, body: { error: "Booking service unavailable" } }
  end

private

  def build_payload(body)
    {
      name:     body["name"],
      zip:      body["zip_code"],
      phone:    body["phone"],
      summary:  body["job_description"].to_s.truncate(200),
      estimate: { min: body["estimate_lo"].to_f.round, max: body["estimate_hi"].to_f.round },
      deadline: body["deadline"].presence || "",
      comment:  "AI Pricing estimate | mid=$#{body['estimate_midpoint'].to_f.round} | #{body['model_version']}",
    }
  end

  def sign(timestamp, payload_body)
    key = ENV.fetch("HA_SIGNING_SECRET", "")
    OpenSSL::HMAC.hexdigest "SHA256", key, "#{timestamp}.#{payload_body}"
  end
end
