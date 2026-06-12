# Style

We love discussing code.
If you have questions about how to write something, or detect some code smell, ask away.
A Pull Request is a great way to do this.

When writing new code, try to find similar code elsewhere to look for inspiration.

## Prefer many files over long files

- Max 200 lines per file
- Max 100 characters per line

When a model exceeds the limit, ask yourself: can I group some methods together into a concern?
When a controller exceeds the limit, ask yourself: would two separate controllers be clearer?
When a helper exceeds the limit, ask yourself: would splitting into multiple helpers be clearer?

## Don't wrap instance variables with accessors unless necessary

Examples:

```ruby
# Bad. Unclear if attribute is used outside of this class.

class User
  attr_reader :name

  def initialize(name)
    @name = name
  end

  def loud_name
    "#{name.upcase}!!!"
  end
end
```

```ruby
# Good. It's clear the attribute is only used within this class.

class User
  def initialize(name)
    @name = name
  end

  def loud_name
    "#{@name.upcase}!!!"
  end
end
```

See more here: https://www.codewithjason.com/dont-wrap-instance-variables-attr_reader-unless-necessary/

## Do not use meta-programming

- Never call private methods using `.send` or `.public_send` on an object.
- Avoid other meta-programming techniques.
- Meta-programming makes code difficult to search and understand.

## Make every database migration reversible

When some actions only happen in the `up` sense, use `up_only`, not `up` and `down`.
When some actions only happen in the `up` and `down` sense, use `reversible`, not `up` and `down`.

```ruby
# Bad
def up
  add_column :bookings, :satisfied, :boolean
  Booking.where(status: :liked).update_all satisfied: true, status: :fulfilled
  Booking.where(status: :disliked).update_all satisfied: false, status: :fulfilled
end

def down
  remove_column :bookings, :satisfied, :boolean
end

# Good
def change
  add_column :bookings, :satisfied, :boolean
  reversible do |dir|
    dir.up do
      Booking.where(status: :liked).update_all satisfied: true, status: :fulfilled
      Booking.where(status: :disliked).update_all satisfied: false, status: :fulfilled
    end
  end
end
```

## Follow REST

We model web endpoints as CRUD operations on resources (REST).
When an action doesn't map cleanly to a standard CRUD verb, we introduce a new resource rather than adding custom actions.

```ruby
# Bad
resources :cards do
  post :close
  post :reopen
end

# Good
resources :cards do
  resource :closure
end
```

## Use Active Record -- do not use Arel or SQL

When writing database operations, always resort to Active Record commands such as `find`,
`insert_all`, `update_all`. Never reach out to Arel commands such as `Arel.sql('count')`
and never reach out for SQL statements.

## Create second-level controllers for nested routes

For example with nested routes:

```ruby
resources :users do
  resources :posts, only: %i[index create], module: :users
end
```

The controller for the `posts` resource should be `Users::PostsController`, not `PostsController`.

## Consider every endpoint to be turbo-enabled

* This means to add `allow_other_host: true` or status: see_other, or data: { turbo_frame: '_top' }


## Limit inline method declarations to these cases:

1. def show; end
2. def advice_params = params.expect booking_advice: [:content]

## Harness the power of nested layouts

- In other words, don't repeat yourself in views
- Even for framework helpers, extract them even with form builders

## Code optimistically

- Never add a method that is not invoked anywhere in the code
- If your code change removes the only usage of a method, remove that method
- Don't add rescue statements for errors that have never happened

## Tests

It's more important to ensure that the feature behaves as expected, rather than
ensuring that a specific method behaves as expected. This encourages us to write
tests that are more resilient to refactoring and less coupled to implementation
details.

- Never write unit tests.
- Never write unit tests for models.
- Prefer integration tests and system tests.

## Use short, meaningful names

Example: Provider, not ServiceProvider. Chat, not Conversation

## Use adjective to name concerns

Example:

```ruby
# Bad
module Booking::AppAssociation extend ActiveSupport::Concern
…
end

# Good
module Booking::Applied extend ActiveSupport::Concern
…
end

```


## Use acronyms when they are correct

Example: API, ZIP

## Do not use single-letter variables

The variable names should be meaningful. Example:

```ruby
# Bad
@vertical.specialties.each do |s|
  @provider.skills.find_or_create_by! specialty: s
end

# Good
@vertical.specialties.each do |specialty|
  @provider.skills.find_or_create_by! specialty: specialty
end
```

## Only use parentheses when needed

Example:

```ruby
# Bad
Skill.find_or_create_by!(specialty: specialty)

# Good
Skill.find_or_create_by! specialty: specialty
```


## For JavaScript, use Stimulus controllers as much as possible

## Visibility modifiers

We indent visibility modifiers at the level of `class` and we don't use special indentation for the
content under them.

```ruby
class SomeClass
  def some_method
    # ...
  end

private

  def some_private_method_1
    # ...
  end

  def some_private_method_2
    # ...
  end
end
```