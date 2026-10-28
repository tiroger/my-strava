# Cost and alerting guardrails.
#
# The account carries unrelated projects, so the budget and anomaly monitor
# are filtered to this application's own tag rather than measuring the whole
# account. Every alarm in the environment publishes to the topic created here.

data "aws_caller_identity" "current" {}

locals {
  name = "my-strava-${var.environment}"
}

resource "aws_sns_topic" "alerts" {
  name = "${local.name}-alerts"
}

# This policy replaces the topic's default one, so it grants every publisher
# the environment uses: Budgets and Cost Explorer for spend, and CloudWatch for
# the worker and queue alarms.
data "aws_iam_policy_document" "alerts" {
  statement {
    sid       = "AllowBudgetsAndCostExplorer"
    effect    = "Allow"
    actions   = ["SNS:Publish"]
    resources = [aws_sns_topic.alerts.arn]

    principals {
      type = "Service"
      identifiers = [
        "budgets.amazonaws.com",
        "costalerts.amazonaws.com",
      ]
    }
  }

  statement {
    sid       = "AllowCloudWatchAlarms"
    effect    = "Allow"
    actions   = ["SNS:Publish"]
    resources = [aws_sns_topic.alerts.arn]

    principals {
      type        = "Service"
      identifiers = ["cloudwatch.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

resource "aws_sns_topic_policy" "alerts" {
  arn    = aws_sns_topic.alerts.arn
  policy = data.aws_iam_policy_document.alerts.json
}

resource "aws_sns_topic_subscription" "email" {
  count = nonsensitive(var.alert_email != null) ? 1 : 0

  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# AWS refuses to activate a cost allocation tag it has not yet seen in billing
# data, and activation is account-wide: if another project in the account has
# already activated the Application key, leave this off.
resource "aws_ce_cost_allocation_tag" "application" {
  count = var.activate_cost_allocation_tag ? 1 : 0

  tag_key = "Application"
  status  = "Active"
}

resource "aws_budgets_budget" "monthly" {
  name         = "${local.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = [format("user:Application$%s", var.application)]
  }

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 50
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_sns_topic_arns = [aws_sns_topic.alerts.arn]
  }

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 90
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_sns_topic_arns = [aws_sns_topic.alerts.arn]
  }

  # Forecast catches a runaway early in the month, when it is still cheap.
  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 100
    threshold_type            = "PERCENTAGE"
    notification_type         = "FORECASTED"
    subscriber_sns_topic_arns = [aws_sns_topic.alerts.arn]
  }

  depends_on = [aws_sns_topic_policy.alerts]
}

resource "aws_ce_anomaly_monitor" "application" {
  name         = "${local.name}-anomaly"
  monitor_type = "CUSTOM"

  # AWS returns user-defined tag keys with a "user:" prefix and the unused
  # sibling expressions as explicit nulls. Declaring anything else makes every
  # plan want to replace the monitor and discard its detection history.
  monitor_specification = jsonencode({
    And            = null
    CostCategories = null
    Dimensions     = null
    Not            = null
    Or             = null
    Tags = {
      Key          = "user:Application"
      Values       = [var.application]
      MatchOptions = ["EQUALS"]
    }
  })
}

resource "aws_ce_anomaly_subscription" "application" {
  name = "${local.name}-anomaly-alerts"

  # SNS subscribers are only accepted at IMMEDIATE frequency.
  frequency = "IMMEDIATE"

  monitor_arn_list = [aws_ce_anomaly_monitor.application.arn]

  subscriber {
    type    = "SNS"
    address = aws_sns_topic.alerts.arn
  }

  threshold_expression {
    dimension {
      key           = "ANOMALY_TOTAL_IMPACT_ABSOLUTE"
      values        = [tostring(var.anomaly_threshold_usd)]
      match_options = ["GREATER_THAN_OR_EQUAL"]
    }
  }

  depends_on = [aws_sns_topic_policy.alerts]
}
