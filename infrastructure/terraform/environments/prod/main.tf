# Production composition.

locals {
  # Non-secret settings are recorded here rather than as CI variables, so
  # plan, apply, and drift detection can never run with different values.

  # Public on the athlete's Strava profile.
  strava_athlete_id = "644338"

  # Set to the ID Strava returns once the webhook is registered against this
  # environment's URL. Strava allows one subscription per API application.
  # Null acknowledges and drops events; the daily reconciliation still runs.
  strava_subscription_id = null

  # Turned true once the repository is connected in the Amplify console with
  # the GitHub App. Terraform then owns the branch and the deploy workflow
  # builds it.
  repository_connected = false

  # If Amplify attached a "CreatedByAmplify-..." Web ACL when the repository
  # was connected, record it here and add an import block, as described in
  # infrastructure/terraform/README.md. Until then its User-Agent rule may
  # refuse Strava's webhook validation.
  amplify_web_acl = null # { name = "CreatedByAmplify-...", metric_name = "Amplify-..." }
}

module "observability" {
  source = "../../modules/observability-baseline"

  environment        = "prod"
  monthly_budget_usd = var.monthly_budget_usd
  alert_email        = var.alert_email
}

module "web" {
  source = "../../modules/web"

  environment             = "prod"
  branch_name             = "production"
  repository_connected    = local.repository_connected
  github_access_token     = var.amplify_github_token
  strava_events_queue_url = module.strava_ingestion.queue_url
  strava_athlete_id       = local.strava_athlete_id
  strava_subscription_id  = local.strava_subscription_id
}

module "strava_ingestion" {
  source = "../../modules/strava-ingestion"

  environment        = "prod"
  source_root        = "${path.root}/../../../.."
  consumer_role_name = module.web.compute_role_name
  alarm_topic_arn    = module.observability.alert_topic_arn
  data_bucket        = module.web.data_bucket
  data_bucket_arn    = module.web.data_bucket_arn
  activities_s3_key  = module.web.activities_s3_key
  bikes_s3_key       = module.web.bikes_s3_key
  strava_athlete_id  = local.strava_athlete_id
}

module "web_firewall" {
  source   = "../../modules/web-firewall"
  for_each = local.amplify_web_acl == null ? {} : { amplify = local.amplify_web_acl }

  web_acl_name = each.value.name
  metric_name  = each.value.metric_name
}
