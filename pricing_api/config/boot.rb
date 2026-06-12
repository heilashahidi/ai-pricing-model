ENV["BUNDLE_GEMFILE"] ||= File.expand_path("../Gemfile", __dir__)

# Load .env from project root if present (dev convenience)
_env_file = File.expand_path("../../.env", __dir__)
if File.exist?(_env_file)
  File.foreach(_env_file) do |line|
    line = line.strip
    next if line.empty? || line.start_with?("#") || !line.include?("=")
    key, _, value = line.partition("=")
    ENV[key.strip] ||= value.strip
  end
end

require "bundler/setup" # Set up gems listed in the Gemfile.
require "bootsnap/setup" # Speed up boot time by caching expensive operations.
