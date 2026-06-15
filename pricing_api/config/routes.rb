Rails.application.routes.draw do
  get "up" => "rails/health#show", as: :rails_health_check

  # Serve demo UI at root
  root to: redirect("/index.html", status: 302)

  # Primary pricing endpoint — mirrors Netlify function path convention
  match "/.netlify/functions/pricing-estimate" => "pricing#estimate", via: :all

  # Public homeowner-facing endpoint — no Bearer required, secret stays server-side
  post "/api/estimate" => "pricing#public_estimate"
  post "/api/book"     => "pricing#book"
end
