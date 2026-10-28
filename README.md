# My Strava

A personal activity dashboard for exploring Strava history, tracking equipment,
and comparing performance with your own previous seasons.

## What it does

- Tracks Strava activities by sport, including cycling, running, and Pickleball.
- Compares monthly distance, active time, and session counts across years using
  Apache ECharts, with tooltips, zoom, and a sport filter.
- Summarizes training volume and personal bests for the selected activity type.
- Shows recent activity, searchable history, GPS route traces, and bike mileage.
- Keeps public viewing separate from maintenance edits, which require an owner
  passphrase.
- Refreshes new and updated activities from Strava webhooks, with a daily
  incremental reconciliation as a safety net.

The new web app is in `apps/web`. The original Streamlit source files remain in
the repository as a migration reference; they are not part of the AWS deployment.

## Run locally

Requires Node.js 22.

```bash
cd apps/web
npm ci
npm run dev
```

Without `APP_DATA_BUCKET`, the app reads the checked-in
`data/processed_data.csv` and `data/bike_data.csv` as a development snapshot.
Local changes to maintenance records require the AWS-backed API and credentials.

Frontend checks:

```bash
cd apps/web
npm run typecheck
npm run lint
npm test
npm run build
```

Worker checks:

```bash
python3 -m unittest tests/test_strava_worker.py
```

## AWS deployment

The app uses AWS Amplify Hosting for the Next.js app, S3 for durable activity
and maintenance data, an SQS FIFO queue for webhook events, Lambda for Strava
API work, and EventBridge for daily reconciliation. It does not use a database
or an always-running application server. The deployment is modelled on
windchaser-ai, which runs the same stack.

Amplify Hosting compute supports Next.js 12 through 15, so the app is pinned to
Next.js 15. Infrastructure is in [`infrastructure/terraform`](./infrastructure/terraform)
and CI/CD in [`.github/workflows`](./.github/workflows):

| Workflow | Trigger | What it does |
|---|---|---|
| CI | pull requests, and before every deploy | secret and state-file check, web and worker tests, Terraform validate |
| Terraform Plan | pull requests touching infrastructure | read-only plan of dev and prod, summarised on the run |
| Deploy development | push to `master` | CI, Terraform apply, Amplify build of that commit, health check |
| Deploy production | push to `production` | the same, behind the `production` environment's approval |
| Terraform Drift | weekly | read-only plan; fails if AWS no longer matches the code |

Amplify's auto-build is off: the deploy workflow starts each build only after
CI and Terraform succeed. Because Amplify exposes environment variables only at
build time, this also means a configuration change reaches the running app on
the same run.

### First-time setup

1. **Bootstrap**, once, with administrator credentials: follow
   [`infrastructure/terraform/bootstrap/README.md`](./infrastructure/terraform/bootstrap/README.md).
   It creates the state bucket and the plan and per-environment apply roles,
   and prints the `gh` commands that set `TERRAFORM_STATE_BUCKET`,
   `AWS_REGION`, `AWS_TERRAFORM_PLAN_ROLE_ARN`, and each environment's
   `AWS_TERRAFORM_APPLY_ROLE_ARN`. Create the `development` and `production`
   GitHub environments first, and add a required reviewer to `production`.
   Optionally `gh secret set ALERT_EMAIL` for alarms, and confirm the email AWS
   sends.
2. **Deploy development**: push to `master` or run **Deploy development**. It
   creates the Amplify app, data bucket, secrets, queues, worker, alarms, and
   budget. The app is not built yet; it has no repository.
3. **Connect the repository** in the Amplify console with the GitHub App
   (Amplify → `my-strava-dev` → connect repository → `tiroger/my-strava`,
   branch `master`). Then set `repository_connected = true` in
   `infrastructure/terraform/environments/dev/main.tf` and push. Terraform takes
   over the branch and the workflow builds and health-checks the app. If the
   console attached a `CreatedByAmplify-...` Web ACL, adopt it as described in
   [`infrastructure/terraform/README.md`](./infrastructure/terraform/README.md),
   or Strava's webhook validation will be refused.
4. **Populate the two secrets** whose ARNs Terraform prints, directly with AWS
   tooling from files kept outside the repository:

   - `strava_secret_arn` (worker only): `client_id`, `client_secret`,
     `refresh_token`. Authorize the token with `activity:read` and
     `profile:read_all`. Avoid `activity:read_all`: the dashboard is public,
     and that scope includes privacy-zone route data. The worker also skips
     "Only You" activities.
   - `web_secret_arn` (web app only): `webhook_verify_token` (at least 16
     characters), `admin_passphrase` (at least 16), and `admin_session_secret`
     (at least 32). Generate them, for example with `openssl rand -base64 32`.

   ```bash
   aws secretsmanager put-secret-value \
     --secret-id <strava-secret-arn> \
     --secret-string file:///secure/path/strava-oauth.json
   aws secretsmanager put-secret-value \
     --secret-id <web-secret-arn> \
     --secret-string file:///secure/path/strava-web.json
   ```

5. **Optionally, queue the initial sync.** With no sync history, the daily
   reconciliation imports the full history on its first run anyway; this only
   saves waiting up to a day. It goes through the queue so it never runs
   alongside another update; follow it in the worker's CloudWatch logs.

   ```bash
   aws sqs send-message \
     --queue-url <strava_events_queue_url> \
     --message-group-id strava-events \
     --message-body '{"source":"scheduled-reconciliation"}'
   ```

6. **Register the Strava webhook** with callback URL
   `<web_url>/api/strava/webhook` and the same `webhook_verify_token`. Record
   the returned ID as `strava_subscription_id` in the environment's `main.tf`
   and push. A Strava API application can have one subscription; if both
   environments need immediate updates, give each its own Strava application.
   Otherwise the other environment relies on the daily reconciliation.

Production follows the same steps against `environments/prod` and the
`production` branch, which you create from `master` when ready to release.

### How events are processed

The worker refreshes OAuth credentials and writes rotated Strava refresh tokens
back to its secret. Strava does not sign webhook payloads, so the endpoint
checks only the subscription and athlete IDs, acknowledges anything it will not
process (so Strava does not retry or count failures against the subscription),
and queues the rest. The worker then confirms every event with Strava before
changing data: it stores an activity only if it belongs to the configured
athlete, and removes one only after Strava reports it gone. One FIFO message
group applies changes one at a time. Failed messages are retried and then dead
lettered, and the dead letter queue, a stalled queue, and worker errors each
raise an alarm. The dashboard checks a lightweight data version every minute
while open and reloads its data only when something changed.

### Cost and privacy

No RDS or other database is required. S3, Lambda, SQS, SNS, and EventBridge are
usage-based, but Amplify Hosting and Secrets Manager can incur charges even at
low traffic, and an adopted Web ACL bills monthly. Each environment has a
budget and an anomaly monitor filtered to the `Application = my-strava` tag;
the `Application` cost allocation tag must be active in the account for them to
see spend.

The dashboard itself is public, as configured for this app. Activity history
and route data are therefore visible to visitors, apart from "Only You"
activities. Routes can reveal where rides start and end; keep Strava privacy
zones set up and the token scoped as described above. Maintenance writes are
protected by a server-verified passphrase and an HTTP-only signed session
cookie. Never expose Strava OAuth credentials or the admin/signing secrets to
browser JavaScript.
