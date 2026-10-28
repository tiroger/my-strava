# Amplify hosting for the Next.js dashboard.
#
# Platform WEB_COMPUTE is what runs server-side rendering and API routes; the
# static platform would drop them, and with them every server-only secret.

locals {
  name = "my-strava-${var.environment}"

  # Set on both the app and the branch. The build sees app-level values, but
  # the SSR compute runtime reads the branch's, and an app-level value alone is
  # invisible to it. amplify.yml then writes these into .env.production, which
  # is what actually carries them into the server bundle.
  runtime_environment = {
    APP_DATA_BUCKET         = aws_s3_bucket.data.id
    ACTIVITIES_S3_KEY       = var.activities_s3_key
    BIKES_S3_KEY            = var.bikes_s3_key
    MAINTENANCE_S3_KEY      = var.maintenance_s3_key
    WEB_SECRET_ARN          = aws_secretsmanager_secret.web.arn
    STRAVA_EVENTS_QUEUE_URL = var.strava_events_queue_url == null ? "" : var.strava_events_queue_url
    STRAVA_ATHLETE_ID       = var.strava_athlete_id == null ? "" : var.strava_athlete_id
    STRAVA_SUBSCRIPTION_ID  = var.strava_subscription_id == null ? "" : var.strava_subscription_id
  }
}

resource "aws_amplify_app" "web" {
  name        = local.name
  description = "Personal Strava training dashboard (${var.environment})."
  platform    = "WEB_COMPUTE"

  # Amplify's CreateApp rejects a repository without a token, so the app is
  # created unconnected and the repository is attached once through the
  # console with the GitHub App. Both are then ignored, so a console
  # connection is never reverted by a later apply.
  repository   = var.github_access_token == null ? null : var.repository_url
  access_token = var.github_access_token

  iam_service_role_arn = aws_iam_role.build.arn
  compute_role_arn     = aws_iam_role.compute.arn

  # No build_spec here: amplify.yml at the repository root takes precedence and
  # is reviewed alongside the code it builds.

  environment_variables = merge(local.runtime_environment, {
    AMPLIFY_MONOREPO_APP_ROOT = var.app_root
    # Diff deploys skip work based on changed paths and get it wrong for a
    # monorepo app that depends on files outside its own root.
    AMPLIFY_DIFF_DEPLOY = "false"
  })

  lifecycle {
    ignore_changes = [
      access_token,
      repository,
    ]
  }
}

# A branch cannot exist before the repository is connected. Once it is,
# Terraform owns the branch, because the console connect flow resets the
# framework to values that disable server-side rendering.
resource "aws_amplify_branch" "tracked" {
  count = var.repository_connected ? 1 : 0

  app_id      = aws_amplify_app.web.id
  branch_name = var.branch_name
  framework   = "Next.js - SSR"
  stage       = var.environment == "prod" ? "PRODUCTION" : "DEVELOPMENT"

  # The deploy workflow starts each build after CI and Terraform succeed, so a
  # failing commit never ships and configuration changes always reach the
  # runtime with a fresh build.
  enable_auto_build = false

  # Never empty: the API rejects an empty map and takes framework and stage
  # down with it.
  environment_variables = local.runtime_environment
}

resource "aws_cloudwatch_log_group" "amplify" {
  name              = "/aws/amplify/${aws_amplify_app.web.id}"
  retention_in_days = var.log_retention_days
}
