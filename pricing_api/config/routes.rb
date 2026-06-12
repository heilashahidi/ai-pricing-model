Rails.application.routes.draw do
  get "up" => "rails/health#show", as: :rails_health_check

  # Primary pricing endpoint — mirrors Netlify function path convention
  post "/.netlify/functions/pricing-estimate" => "pricing#estimate"
  match "/.netlify/functions/pricing-estimate" => "pricing#estimate", via: :all
end
