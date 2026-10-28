output "amplify_app_id" {
  description = "Amplify application ID, for console links and CLI calls."
  value       = module.web.app_id
}

output "amplify_branch_name" {
  description = "Branch the deploy workflow builds; empty until the repository is connected."
  value       = module.web.branch_name == null ? "" : module.web.branch_name
}

output "web_url" {
  description = "URL serving this environment; empty until the repository is connected."
  value       = module.web.branch_url == null ? "" : module.web.branch_url
}

output "data_bucket_name" {
  description = "Private bucket holding activities, bikes, and the maintenance log."
  value       = module.web.data_bucket
}

output "strava_secret_arn" {
  description = "Worker-only Strava OAuth secret to populate with put-secret-value."
  value       = module.strava_ingestion.strava_secret_arn
}

output "web_secret_arn" {
  description = "Web webhook and admin secret to populate with put-secret-value."
  value       = module.web.web_secret_arn
}

output "strava_events_queue_url" {
  description = "Queue for webhook events, reconciliation, and the initial sync."
  value       = module.strava_ingestion.queue_url
}

output "strava_events_message_group_id" {
  description = "FIFO message group to use when queueing a manual sync."
  value       = module.strava_ingestion.message_group_id
}

output "worker_function_name" {
  description = "Strava worker, for its CloudWatch logs."
  value       = module.strava_ingestion.worker_function_name
}

output "alert_topic_arn" {
  description = "SNS topic carrying budget and operational alarms."
  value       = module.observability.alert_topic_arn
}
