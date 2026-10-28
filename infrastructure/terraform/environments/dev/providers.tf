provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Application = "my-strava"
      Environment = "dev"
      ManagedBy   = "terraform"
      Repository  = "my-strava"
    }
  }
}
