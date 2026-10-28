provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Application = "my-strava"
      Environment = "prod"
      ManagedBy   = "terraform"
      Repository  = "my-strava"
    }
  }
}
