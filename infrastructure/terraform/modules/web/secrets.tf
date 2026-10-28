# Terraform creates the container and never the contents; the value is written
# once with the AWS CLI. Strava OAuth credentials live in a separate secret in
# the strava-ingestion module that only the worker can read.

resource "aws_secretsmanager_secret" "web" {
  name                    = "my-strava/${var.environment}/web"
  description             = "Strava webhook verification token and admin session secrets."
  recovery_window_in_days = 7
}
