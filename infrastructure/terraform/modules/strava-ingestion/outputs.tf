output "queue_url" {
  description = "Queue the webhook endpoint and reconciliation schedule write to."
  value       = aws_sqs_queue.events.url
}

output "message_group_id" {
  description = "FIFO message group every producer uses, including a manual sync."
  value       = var.message_group_id
}

output "dead_letter_queue_url" {
  description = "Where events land after repeated processing failures."
  value       = aws_sqs_queue.dead_letter.url
}

output "worker_function_name" {
  description = "Worker applying Strava events and reconciliation."
  value       = aws_lambda_function.worker.function_name
}

output "strava_secret_arn" {
  description = "Worker-only Strava OAuth secret to populate outside Terraform."
  value       = aws_secretsmanager_secret.strava.arn
}
