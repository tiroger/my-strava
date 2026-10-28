provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Application = "my-strava"
      Environment = "bootstrap"
      ManagedBy   = "terraform"
      Repository  = "my-strava"
    }
  }
}
