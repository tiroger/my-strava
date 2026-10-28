# Development composition.

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
  # the GitHub App. Terraform then adopts the branch the console created (see
  # the import below) and the deploy workflow builds it.
  repository_connected = true
  amplify_branch       = "master"

  # If Amplify attached a "CreatedByAmplify-..." Web ACL when the repository
  # was connected, record it here and add an import block, as described in
  # infrastructure/terraform/README.md. Until then its User-Agent rule may
  # refuse Strava's webhook validation.
  amplify_web_acl = null # { name = "CreatedByAmplify-...", metric_name = "Amplify-..." }
}

module "observability" {
  source = "../../modules/observability-baseline"

  environment        = "dev"
  monthly_budget_usd = var.monthly_budget_usd
  alert_email        = var.alert_email
}

module "web" {
  source = "../../modules/web"

  environment             = "dev"
  branch_name             = local.amplify_branch
  repository_connected    = local.repository_connected
  github_access_token     = var.amplify_github_token
  strava_events_queue_url = module.strava_ingestion.queue_url
  strava_athlete_id       = local.strava_athlete_id
  strava_subscription_id  = local.strava_subscription_id
}

module "strava_ingestion" {
  source = "../../modules/strava-ingestion"

  environment            = "dev"
  source_root            = "${path.root}/../../../.."
  consumer_role_name     = module.web.compute_role_name
  alarm_topic_arn        = module.observability.alert_topic_arn
  data_bucket            = module.web.data_bucket
  data_bucket_arn        = module.web.data_bucket_arn
  activities_s3_key      = module.web.activities_s3_key
  bikes_s3_key           = module.web.bikes_s3_key
  streams_s3_prefix      = module.web.streams_s3_prefix
  streams_index_s3_key   = module.web.streams_index_s3_key
  saved_routes_s3_key    = module.web.saved_routes_s3_key
  saved_routes_s3_prefix = module.web.saved_routes_s3_prefix
  strava_athlete_id      = local.strava_athlete_id
}

module "web_firewall" {
  source   = "../../modules/web-firewall"
  for_each = local.amplify_web_acl == null ? {} : { amplify = local.amplify_web_acl }

  web_acl_name = each.value.name
  metric_name  = each.value.metric_name
}


# Connecting the repository in the console creates the branch, and also resets
# the platform to static WEB and the framework to "Web", which disables every
# API route. Adopting the branch here lets the next apply put both back, rather
# than failing to create a duplicate. Once the branch is in state this block
# is a no-op, so it can stay.
import {
  for_each = local.repository_connected ? toset([local.amplify_branch]) : toset([])
  to       = module.web.aws_amplify_branch.tracked[0]
  id       = "${module.web.app_id}/${each.value}"
}
