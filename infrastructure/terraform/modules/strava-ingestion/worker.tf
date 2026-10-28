locals {
  worker_name = "my-strava-${var.environment}-strava-worker"

  # Zipped by Terraform rather than a build step: the worker is pure Python
  # against the standard library and the boto3 the runtime provides, so the
  # artefact cannot differ from the reviewed source.
  worker_sources = {
    for file in fileset("${var.source_root}/services/strava_worker", "**/*.py") :
    "strava_worker/${file}" => "${var.source_root}/services/strava_worker/${file}"
  }
}

data "archive_file" "worker" {
  type        = "zip"
  output_path = "${path.module}/.terraform-build/strava-worker.zip"

  dynamic "source" {
    for_each = local.worker_sources

    content {
      content  = file(source.value)
      filename = source.key
    }
  }
}

data "aws_iam_policy_document" "worker_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "worker" {
  name               = local.worker_name
  assume_role_policy = data.aws_iam_policy_document.worker_assume.json
}

data "aws_iam_policy_document" "worker" {
  statement {
    sid       = "Logs"
    effect    = "Allow"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.worker.arn}:*"]
  }

  statement {
    sid       = "ConsumeEvents"
    effect    = "Allow"
    actions   = ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes"]
    resources = [aws_sqs_queue.events.arn]
  }

  statement {
    sid     = "ReadAndWriteSyncedData"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:PutObject"]
    resources = [
      "${var.data_bucket_arn}/${var.activities_s3_key}",
      "${var.data_bucket_arn}/${var.bikes_s3_key}",
      "${var.data_bucket_arn}/${var.sync_state_s3_key}",
      "${var.data_bucket_arn}/${var.streams_index_s3_key}",
      "${var.data_bucket_arn}/${var.saved_routes_s3_key}",
    ]
  }

  # Streams go when their activity is deleted or made private, and a saved
  # route's profile when the route is.
  statement {
    sid     = "ManageDetailObjects"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = [
      "${var.data_bucket_arn}/${var.streams_s3_prefix}*",
      "${var.data_bucket_arn}/${var.saved_routes_s3_prefix}*",
    ]
  }

  # Without ListBucket, S3 answers a missing object with AccessDenied rather
  # than NoSuchKey and the first sync would fail. It is unconditioned because
  # the 404 check carries no s3:prefix for a condition to match; the bucket
  # holds only this application's data.
  statement {
    sid       = "DistinguishMissingObjects"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [var.data_bucket_arn]
  }

  statement {
    sid       = "ReadAndRotateStravaSecret"
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue", "secretsmanager:PutSecretValue"]
    resources = [aws_secretsmanager_secret.strava.arn]
  }
}

resource "aws_iam_role_policy" "worker" {
  name   = "worker"
  role   = aws_iam_role.worker.id
  policy = data.aws_iam_policy_document.worker.json
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/aws/lambda/${local.worker_name}"
  retention_in_days = var.log_retention_days
}

resource "aws_lambda_function" "worker" {
  function_name = local.worker_name
  role          = aws_iam_role.worker.arn
  handler       = "strava_worker.handler.main"
  runtime       = "python3.12"
  architectures = ["arm64"]

  filename         = data.archive_file.worker.output_path
  source_code_hash = data.archive_file.worker.output_base64sha256

  # The work is waiting on Strava, not computing, so the smallest sensible
  # size is enough.
  timeout     = var.worker_timeout_seconds
  memory_size = 256

  environment {
    variables = {
      APP_DATA_BUCKET      = var.data_bucket
      ACTIVITIES_S3_KEY    = var.activities_s3_key
      BIKES_S3_KEY         = var.bikes_s3_key
      SYNC_STATE_S3_KEY    = var.sync_state_s3_key
      STREAMS_S3_PREFIX    = var.streams_s3_prefix
      STREAMS_INDEX_S3_KEY = var.streams_index_s3_key
      # One backfill batch: comfortably inside both Strava's 15-minute read
      # limit and the worker timeout, which the handler also guards.
      STREAMS_BACKFILL_LIMIT = tostring(var.streams_backfill_limit)
      SAVED_ROUTES_S3_KEY    = var.saved_routes_s3_key
      SAVED_ROUTES_S3_PREFIX = var.saved_routes_s3_prefix
      # Profiles are fetched only for new or edited routes, so this caps the
      # first sync; later runs rarely fetch any.
      SAVED_ROUTES_PROFILE_LIMIT = tostring(var.saved_routes_profile_limit)
      STRAVA_SECRET_ARN          = aws_secretsmanager_secret.strava.arn
      STRAVA_ATHLETE_ID          = var.strava_athlete_id == null ? "" : var.strava_athlete_id
    }
  }

  depends_on = [
    aws_iam_role_policy.worker,
    aws_cloudwatch_log_group.worker,
  ]
}

resource "aws_lambda_event_source_mapping" "events" {
  event_source_arn = aws_sqs_queue.events.arn
  function_name    = aws_lambda_function.worker.arn

  # One message per invocation, so one activity Strava refuses to serve cannot
  # send unrelated events back to the queue with it.
  batch_size = 1
}
