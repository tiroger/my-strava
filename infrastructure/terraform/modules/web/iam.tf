# Two roles, because Amplify has two identities and conflating them grants the
# build far more than it needs:
#
#   iam_service_role_arn  build and deploy
#   compute_role_arn      the SSR runtime, where API routes execute
#
# Only the compute role reaches the data bucket and the web secret.

data "aws_caller_identity" "current" {}

data "aws_iam_policy_document" "amplify_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["amplify.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

resource "aws_iam_role" "build" {
  name               = "${local.name}-amplify-build"
  description        = "Amplify build and deploy role for the my-strava dashboard."
  assume_role_policy = data.aws_iam_policy_document.amplify_assume.json
}

data "aws_iam_policy_document" "build_logs" {
  statement {
    sid    = "WriteOwnLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogGroups",
      "logs:DescribeLogStreams",
    ]
    resources = ["arn:aws:logs:*:${data.aws_caller_identity.current.account_id}:log-group:/aws/amplify/*"]
  }
}

resource "aws_iam_role_policy" "build_logs" {
  name   = "logs"
  role   = aws_iam_role.build.id
  policy = data.aws_iam_policy_document.build_logs.json
}

resource "aws_iam_role" "compute" {
  name               = "${local.name}-amplify-compute"
  description        = "SSR compute role for the my-strava dashboard."
  assume_role_policy = data.aws_iam_policy_document.amplify_assume.json
}

data "aws_iam_policy_document" "compute" {
  statement {
    sid     = "ReadDashboardData"
    effect  = "Allow"
    actions = ["s3:GetObject"]
    resources = [
      "${aws_s3_bucket.data.arn}/${var.activities_s3_key}",
      "${aws_s3_bucket.data.arn}/${var.bikes_s3_key}",
      "${aws_s3_bucket.data.arn}/${var.maintenance_s3_key}",
    ]
  }

  # Without ListBucket, S3 answers a missing object with AccessDenied rather
  # than NoSuchKey, and the dashboard could not tell "not synced yet" apart
  # from a permissions failure.
  statement {
    sid       = "DistinguishMissingObjects"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn]

    condition {
      test     = "StringEquals"
      variable = "s3:prefix"
      values   = [var.activities_s3_key, var.bikes_s3_key, var.maintenance_s3_key]
    }
  }

  statement {
    sid       = "WriteMaintenanceLog"
    effect    = "Allow"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.data.arn}/${var.maintenance_s3_key}"]
  }

  statement {
    sid       = "ReadWebSecret"
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.web.arn]
  }
}

resource "aws_iam_role_policy" "compute" {
  name   = "application-runtime"
  role   = aws_iam_role.compute.id
  policy = data.aws_iam_policy_document.compute.json
}
