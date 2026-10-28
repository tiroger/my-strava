variable "aws_region" {
  description = "AWS region for the production environment."
  type        = string
  default     = "us-east-1"
}

variable "monthly_budget_usd" {
  description = "Monthly production budget threshold in USD."
  type        = number
  default     = 15
}

variable "alert_email" {
  description = <<-EOT
    Address receiving budget and operational alarms. Supplied from the
    ALERT_EMAIL GitHub secret, because the repository is public.
  EOT
  type        = string
  default     = null
  sensitive   = true
}

variable "amplify_github_token" {
  description = <<-EOT
    Optional GitHub token letting Amplify read the repository. Left null, the
    repository is connected once through the console with the GitHub App.
  EOT
  type        = string
  default     = null
  sensitive   = true
}
