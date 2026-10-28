output "alert_topic_arn" {
  description = "SNS topic carrying budget, anomaly, and operational alarms."
  value       = aws_sns_topic.alerts.arn
}

output "budget_name" {
  description = "Name of the monthly budget for this environment."
  value       = aws_budgets_budget.monthly.name
}
