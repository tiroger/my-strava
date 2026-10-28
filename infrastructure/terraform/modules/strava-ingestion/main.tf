# Strava webhook ingestion.
#
# Strava expects the callback to answer within about two seconds, so the web
# endpoint validates, enqueues, and returns. The worker applies each event off
# the queue, confirming it against the Strava API first.
#
# Both the webhook and the daily reconciliation rewrite the same activity
# object, so every message shares one FIFO message group: SQS hands the worker
# one change at a time without capping Lambda concurrency, which would throttle
# messages into the dead letter queue during a long sync.

locals {
  name = "my-strava-${var.environment}-strava-events"
}

resource "aws_sqs_queue" "dead_letter" {
  name                      = "${local.name}-dlq.fifo"
  fifo_queue                = true
  message_retention_seconds = var.event_retention_days * 24 * 60 * 60
  sqs_managed_sse_enabled   = true
}

resource "aws_sqs_queue" "events" {
  name                        = "${local.name}.fifo"
  fifo_queue                  = true
  content_based_deduplication = true
  message_retention_seconds   = var.event_retention_days * 24 * 60 * 60
  sqs_managed_sse_enabled     = true

  # Derived from the worker's timeout: Lambda refuses an event source whose
  # queue could hand a message out again while the function is still running.
  # Six times over is AWS's guidance, and a failing message blocks its FIFO
  # group for this long per attempt.
  visibility_timeout_seconds = var.worker_timeout_seconds * 6

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dead_letter.arn
    maxReceiveCount     = var.max_receive_count
  })
}

# Only the web runtime enqueues: Strava talks to the endpoint, never to SQS.
data "aws_iam_policy_document" "enqueue" {
  statement {
    sid       = "EnqueueWebhookEvents"
    effect    = "Allow"
    actions   = ["sqs:SendMessage"]
    resources = [aws_sqs_queue.events.arn]
  }
}

resource "aws_iam_role_policy" "enqueue" {
  name   = "enqueue-strava-events"
  role   = var.consumer_role_name
  policy = data.aws_iam_policy_document.enqueue.json
}

# The daily reconciliation and the streams backfill enter through the same
# queue and message group, so neither can interleave with a webhook event.
data "aws_iam_policy_document" "scheduled_enqueue" {
  statement {
    effect    = "Allow"
    actions   = ["sqs:SendMessage"]
    resources = [aws_sqs_queue.events.arn]

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }

    condition {
      test     = "ArnEquals"
      variable = "aws:SourceArn"
      values = [
        aws_cloudwatch_event_rule.reconciliation.arn,
        aws_cloudwatch_event_rule.streams_backfill.arn,
      ]
    }
  }
}

resource "aws_sqs_queue_policy" "events" {
  queue_url = aws_sqs_queue.events.url
  policy    = data.aws_iam_policy_document.scheduled_enqueue.json
}

resource "aws_cloudwatch_event_rule" "reconciliation" {
  name                = "${local.name}-reconciliation"
  description         = "Reconcile recent Strava activities as a webhook safety net."
  schedule_expression = var.reconciliation_schedule
  state               = var.schedules_enabled ? "ENABLED" : "DISABLED"
}

resource "aws_cloudwatch_event_target" "reconciliation" {
  rule      = aws_cloudwatch_event_rule.reconciliation.name
  target_id = "strava-event-queue"
  arn       = aws_sqs_queue.events.arn
  input     = jsonencode({ source = "scheduled-reconciliation" })

  sqs_target {
    message_group_id = var.message_group_id
  }

  depends_on = [aws_sqs_queue_policy.events]
}

# Detailed streams cost one Strava call per activity, so history is filled in
# a batch at a time; a full import completes over a few days within the daily
# API allowance.
resource "aws_cloudwatch_event_rule" "streams_backfill" {
  name                = "${local.name}-streams-backfill"
  description         = "Fetch detailed streams for activities that do not have them yet."
  schedule_expression = var.streams_backfill_schedule
  state               = var.schedules_enabled ? "ENABLED" : "DISABLED"
}

resource "aws_cloudwatch_event_target" "streams_backfill" {
  rule      = aws_cloudwatch_event_rule.streams_backfill.name
  target_id = "strava-event-queue"
  arn       = aws_sqs_queue.events.arn
  input     = jsonencode({ source = "streams-backfill" })

  sqs_target {
    message_group_id = var.message_group_id
  }

  depends_on = [aws_sqs_queue_policy.events]
}

# Strava OAuth credentials. Only the worker reads them, and it writes back the
# rotated refresh token Strava may return: an unrecorded rotation would stop
# the worker some days later with nothing pointing at the cause.
resource "aws_secretsmanager_secret" "strava" {
  name                    = "my-strava/${var.environment}/strava-oauth"
  description             = "Strava OAuth client credentials and refresh token."
  recovery_window_in_days = 7
}
