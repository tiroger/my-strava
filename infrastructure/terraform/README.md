# Terraform

All AWS infrastructure for my-strava. The layout and conventions follow
windchaser-ai, which runs the same Amplify, Strava webhook, and OIDC stack.

## Structure

```text
bootstrap/                    Remote state, GitHub OIDC, CI plan and apply roles
environments/dev/             Development composition (branch master)
environments/prod/            Production composition (branch production)
modules/
  observability-baseline/     Alert topic, tag-filtered budget, anomaly monitor
  web/                        Amplify app and branch, data bucket, web secret
  strava-ingestion/           FIFO event queue, worker, reconciliation, alarms
  web-firewall/               Adopts an Amplify-created Web ACL, when one exists
```

Plans and applies run in GitHub Actions through OIDC. Local applies are a
break-glass path, except for `bootstrap`, which is applied once by hand.

Non-secret settings (athlete ID, webhook subscription ID, whether the
repository is connected) are recorded in each environment's `main.tf` locals
rather than CI variables, so plan, apply, and drift detection always agree. The
only CI input is the `ALERT_EMAIL` secret.

## Local validation

```bash
terraform fmt -recursive -check
terraform -chdir=environments/dev init -backend=false
terraform -chdir=environments/dev validate
```

For a local plan against real state, copy `backend.hcl.example` to
`backend.hcl` (ignored by git) and run `terraform init -backend-config=backend.hcl`.

## Adopting the Amplify firewall

Connecting the repository in the Amplify console can attach a Web ACL named
`CreatedByAmplify-...`. Its common rule set refuses Strava's webhook
validation, which carries no User-Agent, and Strava reports only "GET to
callback URL does not return 200". To keep the protection with a narrow
exception for the webhook path:

1. Find the ACL: `aws wafv2 list-web-acls --scope CLOUDFRONT --region us-east-1`.
2. In the environment's `main.tf`, set
   `amplify_web_acl = { name = "<Name>", metric_name = "<VisibilityConfig.MetricName>" }`.
3. Add an import block beside it, and remove it once the apply succeeds:

   ```hcl
   import {
     to = module.web_firewall["amplify"].aws_wafv2_web_acl.amplify
     id = "<Id>/<Name>/CLOUDFRONT"
   }
   ```

The import runs in CI under the apply role, so the adoption is reviewed in the
pull request like any other change.
