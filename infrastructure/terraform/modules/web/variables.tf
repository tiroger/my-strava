variable "environment" {
  description = "Environment this web app serves."
  type        = string

  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "Environment must be dev or prod."
  }
}

variable "repository_url" {
  description = "GitHub repository backing the Amplify app."
  type        = string
  default     = "https://github.com/tiroger/my-strava"
}

variable "branch_name" {
  description = "Branch this environment tracks."
  type        = string
}

variable "app_root" {
  description = "Monorepo path containing the Next.js application. Must match amplify.yml."
  type        = string
  default     = "apps/web"
}

variable "github_access_token" {
  description = <<-EOT
    Optional personal access token granting Amplify read access. Left null,
    the repository is connected once through the console with the GitHub App,
    which is the more secure option and the one Terraform cannot express.
  EOT
  type        = string
  default     = null
  sensitive   = true
}

variable "repository_connected" {
  description = <<-EOT
    True once the repository is attached. Gates the branch, which cannot exist
    before a repository does, and from then on lets Terraform correct what the
    console connect flow changes.
  EOT
  type        = bool
  default     = false
}

variable "log_retention_days" {
  description = "Retention for the app's build and compute logs."
  type        = number
  default     = 14
}

variable "activities_s3_key" {
  description = "Object holding the synced activities."
  type        = string
  default     = "activities.json"
}

variable "bikes_s3_key" {
  description = "Object holding the synced bikes."
  type        = string
  default     = "bikes.json"
}

variable "maintenance_s3_key" {
  description = "Object holding the maintenance log the owner edits."
  type        = string
  default     = "maintenance.json"
}

variable "streams_s3_prefix" {
  description = "Prefix holding each activity's detailed streams, written by the worker."
  type        = string
  default     = "streams/"
}

variable "streams_index_s3_key" {
  description = "Object summarizing synced streams and their effort summaries, written by the worker."
  type        = string
  default     = "streams-index.json"
}

variable "saved_routes_s3_key" {
  description = "Object listing public saved routes, written by the worker."
  type        = string
  default     = "saved-routes.json"
}

variable "saved_routes_s3_prefix" {
  description = "Prefix holding each saved route's elevation profile, written by the worker."
  type        = string
  default     = "saved-routes/"
}

variable "settings_s3_key" {
  description = "Object holding owner-edited settings such as goals."
  type        = string
  default     = "settings.json"
}

variable "strava_events_queue_url" {
  description = "Queue the webhook endpoint enqueues to. Null acknowledges and drops events."
  type        = string
  default     = null
}

variable "strava_athlete_id" {
  description = "Athlete whose webhook events are accepted. Null acknowledges and drops events."
  type        = string
  default     = null
}

variable "strava_subscription_id" {
  description = <<-EOT
    The webhook subscription this environment answers for. Strava does not
    sign payloads, so this is weak evidence and treated as such: the worker
    re-fetches every activity from the API. Null acknowledges and drops events.
  EOT
  type        = string
  default     = null
}
