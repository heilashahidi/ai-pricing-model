Rails.application.routes.draw do
  get "up" => "rails/health#show", as: :rails_health_check

  # Serve demo UI at root
  root to: redirect("/index.html", status: 302)

  # Primary pricing endpoints — mirror Netlify function path convention
  match "/.netlify/functions/pricing-estimate" => "pricing#estimate", via: :all
  match "/.netlify/functions/pricing-outcome"  => "pricing#outcome",  via: :all

  # Public homeowner-facing endpoints — no Bearer required, secret stays server-side
  post "/api/estimate" => "pricing#public_estimate"
  post "/api/outcome"  => "pricing#outcome"
  post "/api/book"     => "pricing#book"
end
