variable "environment" {
  description = "Environment this baseline guards."
  type        = string

  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "Environment must be dev or prod."
  }
}

variable "application" {
  description = "Value of the Application tag identifying this project's spend."
  type        = string
  default     = "my-strava"
}

variable "monthly_budget_usd" {
  description = "Monthly spend threshold for this environment."
  type        = number

  validation {
    condition     = var.monthly_budget_usd > 0
    error_message = "The monthly budget must be greater than zero."
  }
}

variable "anomaly_threshold_usd" {
  description = "Absolute daily anomaly impact that triggers a notification."
  type        = number
  default     = 5
}

variable "alert_email" {
  description = <<-EOT
    Address receiving budget, anomaly, and operational alarms. AWS sends a
    confirmation request that must be accepted before any alert arrives. Null
    creates the topic without a subscriber.
  EOT
  type        = string
  default     = null
  sensitive   = true
}

variable "activate_cost_allocation_tag" {
  description = <<-EOT
    Activate the Application cost allocation tag so the tag-filtered budget can
    see spend. Account-wide, so only one project in the account should own it,
    and AWS must already have seen the tag in billing data (about a day after
    the first tagged spend) or activation fails with "Tag keys not found".
  EOT
  type        = bool
  default     = false
}
