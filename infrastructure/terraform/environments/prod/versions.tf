terraform {
  # S3 native state locking (use_lockfile) requires 1.10 or later.
  required_version = ">= 1.11.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }

  # Partial configuration: bucket, key and region come from -backend-config in
  # CI, or backend.hcl locally. Without this block those flags are accepted and
  # then ignored, and state is written to the runner and thrown away.
  backend "s3" {}
}
