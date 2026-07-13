# Apex — Staging Deployment Guide
## GitHub Actions + Portainer + Cloudflare Tunnel

---

## Overview

This guide covers the complete setup for a staging environment that:

1. Runs CI (lint → unit tests → integration tests) on every push/PR
2. Builds and pushes a Docker image to **GitHub Container Registry (GHCR)** — free for private repos
3. Triggers **Portainer** on your Proxmox server to pull and redeploy the new image
4. Exposes the staging API at `https://staging.yourdomain.com` via **Cloudflare Tunnel** — no open firewall ports

```
Developer pushes to `develop`
        │
        ▼
  GitHub Actions
  ┌──────────────----───────────────────────────┐
  │  1. Lint + type check (ruff, mypy).         │
  │  2. Unit tests (no DB)                      │
  │  3. Integration tests (real Postgres).      │
  │  4. Build Docker image                      │
  │  5. Push → ghcr.io/yourorg/apex-api:staging │
  │  6. POST → Portainer webhook                │
  └─────────────────────----────────────────────┘
        │
        ▼
  Portainer (Proxmox home server)
  ┌─────────────────────────────────────────┐
  │  Pulls new :staging image               │
  │  Restarts apex-api-staging container    │
  │  Runs Alembic migrations                │
  └─────────────────────────────────────────┘
        │
        ▼
  Cloudflare Tunnel
  staging.yourdomain.com → http://api:8000
```

---

## Part 1 — GitHub Actions Concepts

GitHub Actions is GitHub's built-in CI/CD system. It's free for public repos and has 2,000 free minutes/month for private repos.

### Key Concepts

| Term | What it means |
|------|---------------|
| **Workflow** | A YAML file in `.github/workflows/` that defines what to do |
| **Job** | A unit of work in a workflow — runs on a fresh VM |
| **Step** | A single command or action within a job |
| **Action** | A reusable step published on GitHub Marketplace (e.g. `actions/checkout@v4`) |
| **Runner** | The VM that executes your jobs (`ubuntu-latest` = free GitHub-hosted Ubuntu VM) |
| **Secret** | An encrypted variable stored in GitHub, injected into workflows at runtime |
| **Environment** | A named deployment target with its own secrets and protection rules |
| **Service container** | A Docker container spun up alongside your job (used for test databases) |
| **GHCR** | GitHub Container Registry — free Docker image registry at `ghcr.io` |

### How Triggers Work

```yaml
on:
  push:
    branches: [develop]       # Runs when code is pushed to develop
  pull_request:
    branches: [main, develop] # Runs when a PR targets these branches
  workflow_dispatch:          # Adds a "Run workflow" button in GitHub UI
```

### Job Dependencies (`needs`)

Jobs run in parallel by default. Use `needs` to enforce ordering:

```yaml
jobs:
  lint: ...
  unit-tests: ...
  integration-tests:
    needs: [lint]             # Waits for lint to pass first
  build-and-push:
    needs: [unit-tests, integration-tests]  # Waits for both
  deploy-staging:
    needs: [build-and-push]
```

### Conditional Execution (`if`)

```yaml
# Only deploy on actual pushes to develop, not on PRs
if: github.event_name == 'push' && github.ref == 'refs/heads/develop'
```

---

## Part 2 — Repository Setup

### Step 1: File Structure

Add these files to your repository:

```
apex/
├── .github/
│   └── workflows/
│       └── ci-staging.yml          ← the workflow file (provided)
├── docker-compose.yml              ← existing base compose
├── docker-compose.staging.yml      ← new staging override (provided)
└── .env.staging.example            ← template for your server (provided)
```

### Step 2: Enable GitHub Actions

GitHub Actions is enabled by default on all repositories. Simply creating the
`.github/workflows/` directory and pushing a `.yml` file activates it.

Visit: `https://github.com/yourorg/apex/actions` — you'll see the workflow listed.

---

## Part 3 — GitHub Secrets Setup

Secrets are encrypted variables that your workflow can read but nobody can see.
They are **never** logged, even if you `echo` them.

### Where to add secrets

`GitHub repo → Settings → Secrets and variables → Actions → New repository secret`

### Required secrets

#### `PORTAINER_WEBHOOK_URL`

This is the URL GitHub will POST to trigger Portainer to redeploy your stack.

**How to get it:**
1. Open Portainer → Stacks → your staging stack
2. Click **"Stack webhooks"** (or gear icon → Webhooks)
3. Enable the webhook and copy the URL
4. It looks like: `https://portainer.yourdomain.com/api/webhooks/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`

**Why it's a secret:** The URL itself is the authentication token. Anyone with this URL can trigger a redeploy, so keep it private.

#### `GITHUB_TOKEN` — No setup needed!

GitHub automatically provides a `GITHUB_TOKEN` secret to every workflow. It has permission to push images to GHCR for your repository. You do not create this yourself.

### Optional secrets (for Slack/Discord notifications)

You can add `SLACK_WEBHOOK_URL` or `DISCORD_WEBHOOK_URL` later and use them in the notification steps of the workflow.

---

## Part 4 — GitHub Environments Setup

Environments let you add approval gates and environment-specific secrets.

### Create the staging environment

`GitHub repo → Settings → Environments → New environment → Name: staging`

**Recommended settings:**
- **Required reviewers**: leave empty for now (you'd add yourself if you want manual approval before deploys)
- **Wait timer**: 0 minutes
- **Deployment branches**: `develop` only

This is what enables the deployment URL shown in the Actions UI:
```yaml
environment:
  name: staging
  url: https://staging.yourdomain.com
```

---

## Part 5 — Portainer Stack Setup

### Step 1: Create a Personal Access Token in Portainer

`Portainer → your user (top right) → My account → Access tokens → Add access token`

Name it `github-actions`. You'll use this if you want to call the Portainer API directly (alternative to webhooks). For the webhook approach in this guide, you don't need it.

### Step 2: Allow Portainer to pull from GHCR

Your staging image is at `ghcr.io/yourorg/apex-api:staging`. If your repo is private, Portainer needs credentials.

**Create a GitHub Personal Access Token (PAT) for image pulling:**
1. `GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens`
2. New token → name: `portainer-ghcr-pull`
3. Permissions: `Read access to packages`
4. Copy the token

**Add registry credentials to Portainer:**
`Portainer → Registries → Add registry → Custom registry`
- URL: `ghcr.io`
- Username: your GitHub username
- Password: the PAT you just created

### Step 3: Create the Staging Stack in Portainer

`Portainer → Stacks → Add stack`

Choose **"Repository"** (not "Web editor"):
- Name: `apex-staging`
- Repository URL: `https://github.com/yourorg/apex`
- Repository reference: `refs/heads/develop`
- Compose path: `docker-compose.yml`
- Additional paths: `docker-compose.staging.yml`
- Check **"Enable relative path volumes"**
- GitOps update: **enabled** (so Portainer can pull latest on webhook)
- Automatic updates: **disabled** (CI controls deploys)

**Environment variables** — add all values from `.env.staging.example`:

| Variable | Value |
|----------|-------|
| `API_IMAGE` | `ghcr.io/yourorg/apex-api:staging` |
| `POSTGRES_PASSWORD` | `your-strong-password` |
| `JWT_SECRET_KEY` | `your-generated-secret` |
| `R2_ACCOUNT_ID` | `your-account-id` |
| `R2_ACCESS_KEY_ID` | `your-key-id` |
| `R2_SECRET_ACCESS_KEY` | `your-secret` |
| `CLOUDFLARE_TUNNEL_TOKEN` | `your-tunnel-token` |
| `XAI_API_KEY` | `your-key` |

Click **Deploy the stack**.

### Step 4: Get the Portainer Webhook URL

After the stack is deployed:
`Stacks → apex-staging → Stack webhooks → Copy URL`

Add this URL to GitHub Secrets as `PORTAINER_WEBHOOK_URL`.

---

## Part 6 — Cloudflare Tunnel Setup

### Step 1: Create the Tunnel

1. Go to **Cloudflare Zero Trust** → `one.dash.cloudflare.com`
2. `Networks → Tunnels → Create a tunnel`
3. Select **Cloudflared** → name it `proxmox-home`
4. You'll see a token — **copy it** → this is `CLOUDFLARE_TUNNEL_TOKEN` in `.env.staging`
5. Do NOT install cloudflared on the server — the `cloudflared` Docker service in the compose file handles this

### Step 2: Add the Public Hostname

`Tunnels → proxmox-home → Configure → Public Hostnames → Add a public hostname`

| Field | Value |
|-------|-------|
| Subdomain | `staging` |
| Domain | `yourdomain.com` |
| Service Type | `HTTP` |
| URL | `api:8000` (Docker service name + port) |

Cloudflare handles HTTPS/TLS automatically. Your API is now reachable at `https://staging.yourdomain.com`.

### Step 3: (Recommended) Gate staging behind Zero Trust Access

To prevent public access to your staging environment:

`Zero Trust → Access → Applications → Add an application → Self-hosted`
- Application name: `Apex Staging`
- Application domain: `staging.yourdomain.com`
- Create a policy → Allow → Emails → `your@email.com`

Anyone accessing `staging.yourdomain.com` will now be redirected to a Cloudflare login page first. This is free on the Zero Trust free tier.

---

## Part 7 — First Deployment Walkthrough

Once everything is set up:

1. **Push to `develop`** → GitHub Actions triggers automatically

2. **Watch the workflow run:**
   `GitHub → Actions → CI / Deploy Staging → click the run`

   You'll see the job graph: Lint and Unit Tests run in parallel, then Integration Tests,
   then Build & Push, then Deploy.

3. **Build & Push job** logs will show the image being built and pushed to GHCR:
   ```
   Successfully tagged ghcr.io/yourorg/apex-api:staging
   Successfully pushed
   ```

4. **Deploy job** sends a POST to Portainer's webhook URL. Portainer:
   - Pulls `ghcr.io/yourorg/apex-api:staging`
   - Stops the old `apex-api-staging` container
   - Starts the new one
   - The `migrations` service runs `alembic upgrade head` and exits

5. **Health check** — the workflow polls `https://staging.yourdomain.com/health/` until it responds 200

6. **Done.** The staging environment is live.

---

## Part 8 — Makefile Additions

Add these targets to your existing `Makefile`:

```makefile
# =============================================================================
# Staging (local override, for testing the compose setup before CI)
# =============================================================================

staging:
	docker compose --env-file .env.staging \
		-f docker-compose.yml \
		-f docker-compose.staging.yml \
		up -d --build
	@echo "Staging environment started"

staging-down:
	docker compose --env-file .env.staging \
		-f docker-compose.yml \
		-f docker-compose.staging.yml \
		down

staging-logs:
	docker compose --env-file .env.staging \
		-f docker-compose.yml \
		-f docker-compose.staging.yml \
		logs -f

staging-migrate:
	docker compose --env-file .env.staging \
		-f docker-compose.yml \
		-f docker-compose.staging.yml \
		exec api alembic upgrade head
```

---

## Part 9 — Troubleshooting

### Workflow doesn't trigger

- Check the file is at exactly `.github/workflows/ci-staging.yml`
- Check the YAML indentation — GitHub will show parse errors in the Actions tab
- Check the branch name matches (`develop` vs `development`)

### Build fails: "permission denied" pushing to GHCR

- Verify the workflow has `permissions: packages: write`
- Check your repo/org settings allow Actions to write packages:
  `Settings → Actions → General → Workflow permissions → Read and write permissions`

### Portainer webhook returns 4xx

- The webhook URL includes the auth token — if you copied it wrong, get a new one from Portainer
- Make sure the stack name in Portainer matches what the webhook was created for

### Container fails to start — image not found

- Confirm the `API_IMAGE` env var in Portainer matches exactly what was pushed:
  `ghcr.io/yourorg/apex-api:staging` (case-sensitive)
- Confirm Portainer's GHCR registry credentials are correct
- Check Portainer logs: `Stacks → apex-staging → container logs`

### Health check times out

- SSH into Proxmox server and check: `docker logs apex-api-staging`
- Common cause: missing env var (the `:?` syntax will cause the container to fail immediately with a clear error)
- Check the Cloudflare tunnel is online: `Zero Trust → Tunnels → proxmox-home → Status: Healthy`

### `alembic upgrade head` fails

- The `migrations` service exits after running — check its logs in Portainer
- Usually a DATABASE_URL misconfiguration or migration conflict

---

## Summary — Secrets Checklist

| Secret | Where | Description |
|--------|-------|-------------|
| `PORTAINER_WEBHOOK_URL` | GitHub Secrets | Portainer stack webhook URL |
| `GITHUB_TOKEN` | Auto-provided | Pushes image to GHCR — no setup needed |
| All `.env.staging` vars | Portainer Stack env | Runtime config for containers |
| `CLOUDFLARE_TUNNEL_TOKEN` | Portainer Stack env | Cloudflare tunnel auth |
| GHCR PAT | Portainer Registries | For Portainer to pull the image |
