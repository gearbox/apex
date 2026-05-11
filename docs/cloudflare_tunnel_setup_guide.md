# Cloudflare Tunnel Setup Guide: Apex ↔ Aisha over `gpu.gpu-sessions-domain.com`

## Architecture overview

There are two completely separate tunnel concerns in this architecture.
Getting them confused is the most common mistake:

|                       | **Apex API tunnel**                                        | **Per-session GPU node tunnel**                             |
| --------------------- | ---------------------------------------------------------- | ----------------------------------------------------------- |
| **What**              | Exposes the staging apex API publicly                      | Connects each ephemeral Vast.ai GPU node back to CF         |
| **Created by**        | You, manually, once                                        | Apex Python code, automatically, at session start           |
| **Token lives in**    | Staging docker-compose (`CLOUDFLARE_TUNNEL_TOKEN`)         | GPU node env (`ACS_CF_TUNNEL_TOKEN`), rotated per session   |
| **Hostname**          | `staging-api.apex-domain.com`                                     | `{session-short-id}.gpu.gpu-sessions-domain.com`                      |
| **Lifetime**          | Permanent                                                  | Ephemeral — deleted when the session ends                   |

---

## Phase 1 — Prerequisites: collect your IDs

You need three identifiers from the CF dashboard before doing anything.
Log in to [dash.cloudflare.com](https://dash.cloudflare.com).

**Account ID**
Right sidebar of the dashboard home → copy **Account ID**.
Looks like `699d98642c564d2e855e9661899b7252`.

**Zone ID for `apex-domain.com`**
**Websites** → click `apex-domain.com` → **Overview** tab → right sidebar → **Zone ID**.
Looks like `9f5890f86b09f8b7b6a8cd5c935c8bc7`.

**Zone ID for `gpu-sessions-domain.com`**
**Websites** → click `gpu-sessions-domain.com` → **Overview** tab → right sidebar → **Zone ID**.
Different value from the `apex-domain.com` Zone ID — save both.

Save all three somewhere; you will need them in multiple steps below.

---

## Phase 2 — Create the apex API's own tunnel (one-time, manual)

This is the always-on tunnel that makes `staging-api.apex-domain.com` reachable
from the internet. Your other projects already have tunnels — this is the
same process, just for apex.

### 2.1 Create the tunnel

1. Go to [one.dash.cloudflare.com](https://one.dash.cloudflare.com) →
   **Networks** → **Connectors** → **Cloudflare Tunnels**
2. Click **Create a tunnel**
3. Choose **Cloudflared** → **Next**
4. Name it `apex-staging` → **Save tunnel**
5. CF shows you a `cloudflared service install --token eyJ...` command.
   **Copy only the `eyJ...` token** — that is your `CLOUDFLARE_TUNNEL_TOKEN`
   for the staging docker-compose. Do not run the install command here;
   your tunnel runs inside Docker as the `cloudflared` service, not as a
   system service.
6. Click **Next** → skip the "Published applications" step for now
   (you will configure it in 2.2)

### 2.2 Add the published application route

Still in the tunnel wizard, or via **Edit** on the tunnel afterward:

1. Go to the **Published applications** tab → **Add a route**
2. **Subdomain**: `staging-api`, **Domain**: `apex-domain.com`
   (this produces the hostname `staging-api.apex-domain.com`)
3. **Service type**: `HTTP`, **URL**: `http://api:8100`
   (the apex API's internal docker hostname and port)
4. Save

CF automatically creates a proxied CNAME:
`staging-api.apex-domain.com → <tunnel-uuid>.cfargotunnel.com`

No manual DNS record needed — CF handles it.

### 2.3 Wire the token into your staging compose

In your `.env.staging` or Portainer secrets:

```
CLOUDFLARE_TUNNEL_TOKEN=eyJ...   ← the token from step 2.1
```

The `cloudflared` service in `docker-compose.staging.yml` already reads
this variable. Once deployed it establishes the outbound connection and
`staging-api.apex-domain.com` becomes reachable.

---

## Phase 3 — Create the API token for per-session GPU tunnels

This is the critical piece that is easy to overlook. Apex needs a
Cloudflare **API token** (distinct from a tunnel token) so it can call
the Cloudflare API at runtime to create a new tunnel UUID for each GPU
session, and to create/delete the matching DNS record under
`gpu.gpu-sessions-domain.com`.

### Required permissions on this token

| Permission             | Level                              | Why                                                    |
| ---------------------- | ---------------------------------- | ------------------------------------------------------ |
| `Cloudflare Tunnel:Edit` | Account                          | Create and delete tunnels programmatically             |
| `DNS:Edit`             | Zone — **`gpu-sessions-domain.com` only**    | Create `{id}.gpu.gpu-sessions-domain.com` CNAME per session      |

Note: `apex-domain.com` DNS edit is **not** needed here. This token only touches
`gpu-sessions-domain.com` for the GPU subdomain records.

### 3.1 Create the token

1. [dash.cloudflare.com](https://dash.cloudflare.com) → top-right avatar
   → **My Profile** → **API Tokens**
2. **Create Token** → **Create Custom Token**
3. Name it `apex-gpu-session-manager`
4. Add two permission rows:
   - **Account** → **Cloudflare Tunnel** → **Edit**
   - **Zone** → **DNS** → **Edit**
5. Under **Zone Resources**: set to **Specific zone** → `gpu-sessions-domain.com`
   (scopes DNS edit to this zone only, not `apex-domain.com` or any other zone)
6. *(Optional but recommended)* **IP Address Filtering** → add your
   Hetzner server's IP to restrict where this token can be used from
7. **Continue to summary** → **Create Token**
8. **Copy the token immediately** — it is shown only once

### 3.2 Populate apex's staging env

You now have everything for the four `AISHA_CF_*` variables:

```bash
AISHA_CF_API_TOKEN=<the token from 3.1>
AISHA_CF_ACCOUNT_ID=<Account ID from Phase 1>
AISHA_CF_ZONE_ID=<Zone ID for gpu-sessions-domain.com from Phase 1>
AISHA_CF_TUNNEL_DOMAIN=gpu.gpu-sessions-domain.com
```

Add these to your staging `.env` / Portainer secrets and redeploy.

---

## Phase 4 — Prepare `gpu.gpu-sessions-domain.com` for dynamic per-session CNAMEs

Each GPU session gets a subdomain like `a3f9b2.gpu.gpu-sessions-domain.com`. Apex
creates the CNAME via API at session start and deletes it at session end.
You need to verify nothing is blocking this.

### 4.1 Check for conflicting records

[dash.cloudflare.com](https://dash.cloudflare.com) → **Websites** →
`gpu-sessions-domain.com` → **DNS** → **Records**:

- Search for `gpu` — confirm there is no existing A, CNAME, or wildcard
  (`*.gpu`) record
- If nothing is found, you are clear. The subdomain does not need a
  "parent" A record; per-session CNAMEs are created directly by apex

### 4.2 Decide on SSL for `*.gpu.gpu-sessions-domain.com`

When apex creates `a3f9b2.gpu.gpu-sessions-domain.com`, that is a two-level
subdomain. CF's Universal SSL only covers `*.gpu-sessions-domain.com` (one level
deep). A `*.gpu.gpu-sessions-domain.com` wildcard cert is **not** included.

**Option A — Order an Advanced Certificate (cleanest)**
- **SSL/TLS** → **Edge Certificates** → **Advanced Certificate Manager**
  → **Order Advanced Certificate**
- Add `*.gpu.gpu-sessions-domain.com` as a hostname
- Cost: ~$10/month or included in CF Pro/Business
- All per-session subdomains are covered automatically with no further
  action

**Option B — Use HTTP between CF edge and the GPU node (recommended for this use case)**
- When apex creates the per-session tunnel route, set service type to
  `HTTP` pointing at `http://localhost:18188` on the node
- CF still presents HTTPS to apex (the only caller — no browser is
  involved), and the CF-to-node leg is HTTP over the outbound
  `cloudflared` tunnel, which is never exposed to the public internet
- No certificate cost, no extra configuration
- **This is the right architectural choice here**: the ComfyUI endpoint
  is an internal API polled only by apex's Aisha poller, not a public
  website. Users never touch this URL directly.

Use **Option B** unless you have a specific reason to require TLS
on the CF→node leg.

### 4.3 Confirm `gpu-sessions-domain.com` nameservers

You confirmed it is already in your CF profile. Quick sanity check:

```bash
dig NS gpu-sessions-domain.com +short
# Expected: *.ns.cloudflare.com
```

If the output shows Cloudflare nameservers, all DNS changes apex makes
via the API are authoritative and take effect immediately.

---

## Phase 5 — How apex uses this at runtime

For completeness, so you can verify the code path against what you just
set up.

**Session start** (`start_session` in `service.py`):

1. Apex calls `cloudflare_client.create_session_tunnel(session_id)` which
   POSTs to:
   ```
   https://api.cloudflare.com/client/v4/accounts/{AISHA_CF_ACCOUNT_ID}/cfd_tunnel
   ```
   with the session name (e.g. `gpu-{short-id}`) and `config_src: cloudflare`.
   Response contains a tunnel UUID and an `eyJ...` tunnel token.

2. Apex calls the DNS API to create a proxied CNAME:
   ```
   {short-id}.gpu.gpu-sessions-domain.com → {tunnel-uuid}.cfargotunnel.com
   ```

3. Apex passes `ACS_CF_TUNNEL_TOKEN={token}` into the Vast.ai instance
   env alongside the rest of the `ACS_*` variables.

4. The GPU node's `onstart.sh` runs. Supervisord starts:
   ```
   cloudflared tunnel --no-autoupdate run
   ```
   with `TUNNEL_TOKEN` set in the supervisord environment. Cloudflared
   phones home to CF using the token. The tunnel becomes `Healthy`.

5. Apex's Aisha poller can now reach:
   ```
   http://{short-id}.gpu.gpu-sessions-domain.com/system_stats
   ```
   and ComfyUI's job submission endpoint on the same host.

**Session end** (cleanup worker):

1. Apex calls `cloudflare_client.delete_tunnel(tunnel_id)` — deletes the
   tunnel object, which disconnects cloudflared on the node.
2. Apex calls the DNS API to delete the `{short-id}.gpu.gpu-sessions-domain.com`
   CNAME.
3. The Vast.ai instance is terminated.

---

## Phase 6 — Verification checklist

Complete these checks after phases 1–4, before attempting a real GPU
session.

### Apex API tunnel

- [ ] `CLOUDFLARE_TUNNEL_TOKEN` set in staging → redeploy → tunnel
      `apex-staging` shows **Healthy** in
      **Zero Trust → Networks → Cloudflare Tunnels**
- [ ] `curl https://staging-api.apex-domain.com/health` (or your apex health
      endpoint) returns 200
- [ ] DNS record for `staging-api.apex-domain.com` in **apex-domain.com** zone shows
      a proxied CNAME to `<uuid>.cfargotunnel.com`

### API token (per-session tunnel management)

- [ ] Tunnel create permission — from your Hetzner server:
  ```bash
  curl -s "https://api.cloudflare.com/client/v4/accounts/$AISHA_CF_ACCOUNT_ID/cfd_tunnel?per_page=1" \
    -H "Authorization: Bearer $AISHA_CF_API_TOKEN" | jq .success
  # Expected: true
  ```
- [ ] DNS edit permission on `gpu-sessions-domain.com` — create a test record then
      immediately delete it:
  ```bash
  # Create
  curl -s -X POST \
    "https://api.cloudflare.com/client/v4/zones/$AISHA_CF_ZONE_ID/dns_records" \
    -H "Authorization: Bearer $AISHA_CF_API_TOKEN" \
    -H "Content-Type: application/json" \
    --data '{"type":"CNAME","name":"test-setup.gpu.gpu-sessions-domain.com",
             "content":"test.cfargotunnel.com","proxied":true}' | jq .success
  # Expected: true

  # Then delete the record — grab the id from the create response
  # or find it in the CF dashboard DNS tab and remove it
  ```

### `gpu.gpu-sessions-domain.com` readiness

- [ ] No conflicting A / CNAME / wildcard records under `gpu` in
      `gpu-sessions-domain.com` DNS
- [ ] SSL decision made and implemented:
      either Advanced Cert ordered for `*.gpu.gpu-sessions-domain.com`,
      or apex's tunnel-route creation code uses service type `HTTP`
- [ ] All four `AISHA_CF_*` env vars set in staging, apex redeployed,
      no startup errors about CF not configured in logs

---

## Common gotchas

**Tunnel is `Inactive` after cloudflared starts on the GPU node**
The token is wrong or stale. The token is per-tunnel UUID; if apex
deletes and recreates the tunnel during a provisioning retry, the old
token is invalid. The code must always use the token returned from the
*current* `create_tunnel` call, never a cached one.

**CNAME resolves but returns 523 (origin unreachable)**
The `cloudflared` process on the node has not connected yet. Normal
during the ~30–60s node boot window. The Aisha poller's readiness
probe (polling `/system_stats`) retries until the tunnel is healthy or
times out — no manual intervention needed.

**DNS API returns 403 on CNAME creation**
The token's Zone Resources scope is wrong. Verify it is set to
**Specific zone → `gpu-sessions-domain.com`** and not to a different zone or
left as "All zones" (which can fail on accounts where zone-level
permissions conflict with account-level ones).

**`staging-api.apex-domain.com` returns 521 (web server down)**
The apex `api` container is not running or is listening on a different
port. The tunnel is healthy but the upstream `http://api:8100` is
unreachable inside the Docker network. Check container status and that
the port matches what apex actually binds to.

**Existing `apex-domain.com` projects are unaffected**
The apex API tunnel only adds a CNAME for `staging-api.apex-domain.com`. All
other records on `apex-domain.com` are untouched. Per-session GPU tunnels
are entirely on `gpu-sessions-domain.com`, not on `apex-domain.com`.

**Two-level subdomain SSL error when testing in a browser**
Expected if you chose Option B (HTTP service type). Since apex's
`httpx` client calls the GPU tunnel URL directly (not a browser), this
is not an issue in practice. If you do need to inspect a session's
ComfyUI in a browser during debugging, temporarily enable Option A
or use `curl -k` (insecure) for that one-off case.
