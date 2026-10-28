variable "environment" {
  description = "Environment this ingestion path serves."
  type        = string

  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "Environment must be dev or prod."
  }
}

variable "source_root" {
  description = "Repository root, from which the worker's Python is zipped."
  type        = string
}

variable "consumer_role_name" {
  description = "Role permitted to enqueue webhook events: the web runtime."
  type        = string
}

variable "alarm_topic_arn" {
  description = "Topic receiving dead-letter, staleness, and worker error alarms."
  type        = string
}

variable "data_bucket" {
  description = "Bucket holding the synced activities and bikes."
  type        = string
}

variable "data_bucket_arn" {
  description = "ARN of the data bucket, for the worker's object permissions."
  type        = string
}

variable "activities_s3_key" {
  description = "Object holding the synced activities."
  type        = string
}

variable "bikes_s3_key" {
  description = "Object holding the synced bikes."
  type        = string
}

variable "sync_state_s3_key" {
  description = "Object recording when reconciliation last ran."
  type        = string
  default     = "sync-state.json"
}

variable "strava_athlete_id" {
  description = "Athlete whose activities the worker accepts from webhook events."
  type        = string
  default     = null
}

variable "message_group_id" {
  description = "FIFO message group shared by every producer. Must match the web app's EVENTS_MESSAGE_GROUP."
  type        = string
  default     = "strava-events"
}

variable "reconciliation_schedule" {
  description = "How often to reconcile recent activities as a webhook safety net."
  type        = string
  default     = "rate(1 day)"
}

variable "max_receive_count" {
  description = "Deliveries attempted before an event moves to the dead letter queue."
  type        = number
  default     = 5
}

variable "event_retention_days" {
  description = "How long unprocessed events survive in the queues."
  type        = number
  default     = 14

  validation {
    condition     = var.event_retention_days >= 1 && var.event_retention_days <= 14
    error_message = "SQS retains messages for between 1 and 14 days."
  }
}

variable "worker_timeout_seconds" {
  description = <<-EOT
    How long the worker may run. A full first sync pages through the whole
    history; webhook events finish in seconds. The queue's visibility timeout
    is derived from this.
  EOT
  type        = number
  default     = 300

  validation {
    condition     = var.worker_timeout_seconds >= 30 && var.worker_timeout_seconds <= 900
    error_message = "Use between 30 and 900 seconds."
  }
}

variable "log_retention_days" {
  description = "Retention for the worker's logs."
  type        = number
  default     = 14
}
