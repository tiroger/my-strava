# A dead letter queue nobody looks at is just a slower way to lose data.
resource "aws_cloudwatch_metric_alarm" "dead_letter_depth" {
  alarm_name        = "${local.name}-dlq-not-empty"
  alarm_description = "Strava events failed processing repeatedly and need inspection."

  namespace   = "AWS/SQS"
  metric_name = "ApproximateNumberOfMessagesVisible"
  dimensions  = { QueueName = aws_sqs_queue.dead_letter.name }

  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  alarm_actions = [var.alarm_topic_arn]
  ok_actions    = [var.alarm_topic_arn]
}

# A failing FIFO message blocks every event behind it while it is retried, so
# an old head of queue matters as much as a failure does.
resource "aws_cloudwatch_metric_alarm" "events_stale" {
  alarm_name        = "${local.name}-not-draining"
  alarm_description = "Strava events are queued but not being processed."

  namespace   = "AWS/SQS"
  metric_name = "ApproximateAgeOfOldestMessage"
  dimensions  = { QueueName = aws_sqs_queue.events.name }

  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 2
  threshold           = 3600
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  alarm_actions = [var.alarm_topic_arn]
  ok_actions    = [var.alarm_topic_arn]
}

resource "aws_cloudwatch_metric_alarm" "worker_errors" {
  alarm_name        = "${local.worker_name}-errors"
  alarm_description = "The Strava worker failed; SQS retries before dead-lettering."

  namespace   = "AWS/Lambda"
  metric_name = "Errors"
  dimensions  = { FunctionName = aws_lambda_function.worker.function_name }

  statistic           = "Sum"
  period              = 3600
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  alarm_actions = [var.alarm_topic_arn]
  ok_actions    = [var.alarm_topic_arn]
}
