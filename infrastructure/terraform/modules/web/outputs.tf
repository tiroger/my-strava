output "app_id" {
  description = "Amplify application ID."
  value       = aws_amplify_app.web.id
}

output "branch_name" {
  description = "Tracked branch, or null until the repository is connected."
  value       = length(aws_amplify_branch.tracked) > 0 ? aws_amplify_branch.tracked[0].branch_name : null
}

output "branch_url" {
  description = "URL serving the tracked branch, once a branch exists."
  value = length(aws_amplify_branch.tracked) > 0 ? (
    "https://${aws_amplify_branch.tracked[0].branch_name}.${aws_amplify_app.web.default_domain}"
  ) : null
}

output "compute_role_name" {
  description = "SSR compute role, for attaching the enqueue permission."
  value       = aws_iam_role.compute.name
}

output "web_secret_arn" {
  description = "Webhook and admin secret to populate outside Terraform."
  value       = aws_secretsmanager_secret.web.arn
}

output "data_bucket" {
  description = "Bucket holding activities, bikes, and the maintenance log."
  value       = aws_s3_bucket.data.id
}

output "data_bucket_arn" {
  description = "ARN of the data bucket, for the worker's object permissions."
  value       = aws_s3_bucket.data.arn
}

output "activities_s3_key" {
  description = "Object holding the synced activities."
  value       = var.activities_s3_key
}

output "bikes_s3_key" {
  description = "Object holding the synced bikes."
  value       = var.bikes_s3_key
}

output "streams_s3_prefix" {
  description = "Prefix holding each activity's detailed streams."
  value       = var.streams_s3_prefix
}

output "streams_index_s3_key" {
  description = "Object summarizing synced streams."
  value       = var.streams_index_s3_key
}

output "saved_routes_s3_key" {
  description = "Object listing public saved routes."
  value       = var.saved_routes_s3_key
}

output "saved_routes_s3_prefix" {
  description = "Prefix holding each saved route's elevation profile."
  value       = var.saved_routes_s3_prefix
}
