# GPU Node Provisioning Architecture — Apex ↔ Aisha Integration

## 1. Overview

This document describes the architecture for on-demand GPU node provisioning via Vast.ai, integrating the Apex API backend with the Aisha deployment system. Users request a model (e.g., `AISHA_VIDEO`), Apex provisions a Vast.ai node, Aisha deploys the matching bundle, and the node becomes available for generation jobs — all automated, with secure connectivity via Cloudflare Tunnels.

### Scope

- **Phase 1**: Apex-driven provisioning with polling-based readiness detection + CF Tunnel networking
- **Phase 2** (future): Callback-based readiness signaling from GPU node → Apex (env vars pre-wired in Phase 1)
- **Out of scope**: Dynamic pricing (deferred), multi-session per ModelType (YAGNI), WebSocket streaming, session warm pool (deferred), bundle hot-swap (not needed)

### Key Participants

| System | Role |
|--------|------|
| **Apex** | REST API — accepts session requests, calls Vast.ai + Cloudflare APIs, monitors provisioning, routes generation jobs |
| **Aisha** | CLI tool on GPU node — deploys bundles (ComfyUI + models + workflows) via `onstart.sh` |
| **ai-bundles** | Private repo — bundle configs with hardware requirements, model mappings |
| **Vast.ai** | GPU marketplace — provides on-demand GPU instances via REST API |
| **Cloudflare** | Tunnel API — creates secure tunnels for Apex ↔ GPU node connectivity via `gpu-domain.com` |

---

## 2. High-Level Flow

```
User                     Apex                        Cloudflare            Vast.ai              GPU Node
 │                        │                             │                    │                     │
 │  POST /v1/sessions     │                             │                    │                     │
 │  {model: aisha-video}  │                             │                    │                     │
 │───────────────────────>│                             │                    │                     │
 │                        │                             │                    │                     │
 │                        │  1. Resolve model → bundle  │                    │                     │
 │                        │  2. Read hardware reqs      │                    │                     │
 │                        │  3. Reserve billing tokens  │                    │                     │
 │                        │                             │                    │                     │
 │                        │  POST /tunnels              │                    │                     │
 │                        │────────────────────────────>│                    │                     │
 │                        │  ← tunnel_id + token        │                    │                     │
 │                        │                             │                    │                     │
 │                        │  POST /dns_records (CNAME)  │                    │                     │
 │                        │────────────────────────────>│                    │                     │
 │                        │                             │                    │                     │
 │                        │  POST /bundles (search)     │                    │                     │
 │                        │─────────────────────────────────────────────────>│                     │
 │                        │  ← offer_id, dph_total      │                    │                     │
 │                        │                             │                    │                     │
 │                        │  PUT /asks/{offer_id}       │                    │                     │
 │                        │  (create instance + env)    │                    │                     │
 │                        │─────────────────────────────────────────────────>│                     │
 │                        │  ← instance_id              │                    │                     │
 │                        │                             │                    │  instance boots     │
 │  ← 202 Accepted        │                             │                    │────────────────────>│
 │  {session_id, pending}  │                             │                    │                     │
 │                        │                             │                    │  onstart.sh runs    │
 │                        │                             │                    │  - install cloudflared
 │                        │                             │                    │  - start tunnel     │
 │                        │                             │                    │  - clone aisha      │
 │                        │                             │                    │  - clone ai-bundles │
 │                        │                             │                    │  - deploy bundle    │
 │                        │                             │                    │  - ComfyUI ready    │
 │                        │                             │                    │                     │
 │                        │  Provisioning Worker polls  │                    │                     │
 │                        │  https://gpu-{session}.gpu-domain.com/object_info │                     │
 │                        │────────────────────────────>│───────────(tunnel)─────────────────────>│
 │                        │  ← 200 OK                   │                    │                     │
 │                        │                             │                    │                     │
 │                        │  UPDATE gpu_sessions        │                    │                     │
 │                        │  status = 'active'          │                    │                     │
 │                        │  started_at = NOW()         │                    │                     │
 │                        │                             │                    │                     │
 │  SSE: session.active   │                             │                    │                     │
 │<───────────────────────│                             │                    │                     │
```

---

## 3. Bundle System Integration

### 3.1 Bundle → ModelType Mapping

Users request sessions by `ModelType` (e.g., `AISHA_VIDEO`). Apex resolves this to a concrete bundle name. The mapping lives in `bundle-index.yaml` in the `ai-bundles` repo:

```yaml
# ai-bundles/bundle-index.yaml
version: "2"  # bumped to signal new fields

bundles:
  - name: wan_2.2_i2v
    path: bundles/wan_2.2_i2v
    description: "WAN 2.2 Image-to-Video with GGUF Q8 models"
    tags: [video, i2v, wan, gguf]
    model_type: aisha-video          # ← maps to ModelType.AISHA_VIDEO
    default_bundle: true             # ← default bundle for this model_type

  - name: sdxl_base
    path: bundles/sdxl_base
    description: "SDXL base image generation"
    tags: [image, t2i, sdxl]
    model_type: aisha-image
    default_bundle: true

  - name: flux_schnell
    path: bundles/flux_schnell
    description: "Flux Schnell fast image generation"
    tags: [image, t2i, flux]
    model_type: aisha-image
    default_bundle: false            # ← not default, available via admin override
```

**Resolution logic in Apex:**
1. User requests `model: "aisha-video"`
2. Apex scans bundle index for entries where `model_type == "aisha-video"` AND `default_bundle == true`
3. Returns the single default bundle name + reads its `bundle.yaml` for hardware requirements
4. Admin override: `POST /v1/sessions { "model": "aisha-video", "bundle_override": "wan_2.2_i2v:260105-01" }` bypasses the default resolution

### 3.2 Hardware Requirements in `bundle.yaml`

Each bundle declares its GPU requirements in a structured `hardware` section:

```yaml
# ai-bundles/bundles/wan_2.2_i2v/260105-01/bundle.yaml
metadata:
  name: wan_2.2_i2v
  version: "260105-01"
  description: "WAN 2.2 Image-to-Video with GGUF Q8 models - optimized for 24GB VRAM"
  created_at: "2026-01-05T10:30:00Z"
  tested: true
  tags: [video, i2v, wan, gguf]

hardware:
  gpu_whitelist:             # Vast.ai gpu_name values (use "in" operator)
    - RTX_4090
    - RTX_A6000
    - A100_SXM4
    - A100_PCIE
    - H100_SXM
    - H100_PCIE
  min_vram_gb: 16            # Reference only (gpu_whitelist is authoritative)
  recommended_vram_gb: 24    # Reference only
  min_disk_gb: 80            # Storage for models + ComfyUI + workspace
  min_network_upload_mbps: 200    # Minimum upload speed (Mbps)
  min_network_download_mbps: 500  # Minimum download speed (Mbps) — critical for model downloads
  cuda_min_version: "12.1"        # Minimum CUDA version
  num_gpus: 1                     # Number of GPUs required

comfyui:
  repo: https://github.com/comfyanonymous/ComfyUI
  commit: a1b2c3d4e5f6...
  port: 8188                      # ComfyUI listen port inside container

# ... rest of bundle.yaml (custom_nodes, models, workflow_file, etc.)
```

### 3.3 Apex Bundle Registry Client

Apex needs read-only access to `ai-bundles` to resolve mappings and hardware requirements. A lightweight `BundleIndexService` syncs the repo periodically:

```
src/api/services/
└── bundle_index.py    # BundleIndexService — git sync + index/yaml parsing
```

**Sync strategy:**
- On startup: shallow clone (`--depth 1`) of `ai-bundles` to a local cache directory
- Periodic sync: `git pull` every N minutes (configurable via `AI_BUNDLES_SYNC_INTERVAL_MINUTES`, default: 15)
- In-memory cache of parsed `BundleIndex` + per-bundle `HardwareRequirements` dataclass
- Invalidation: any `git pull` that changes files triggers re-parse

**Data structures (Apex-side):**

```python
# src/core/bundle_config.py

@dataclass(frozen=True, slots=True)
class HardwareRequirements:
    """GPU hardware requirements parsed from bundle.yaml."""
    gpu_whitelist: tuple[str, ...]
    min_disk_gb: int
    min_network_upload_mbps: int
    min_network_download_mbps: int
    cuda_min_version: str
    num_gpus: int
    comfyui_port: int = 8188

@dataclass(frozen=True, slots=True)
class BundleMapping:
    """Resolved bundle for a ModelType."""
    bundle_name: str
    bundle_version: str | None  # None = use "current" symlink
    hardware: HardwareRequirements
```

---

## 4. Vast.ai API Integration

### 4.1 Client Design

A thin async client wrapping the Vast.ai REST API:

```
src/api/services/
└── vastai/
    ├── __init__.py
    ├── client.py         # VastAIClient — search offers, create/get/destroy/stop/start instance
    ├── schemas.py        # msgspec Structs for Vast.ai API request/response shapes
    └── exceptions.py     # VastAIError, NoCapacityError, VastAIPaymentError
```

**Key operations:**

| Method | Vast.ai Endpoint | Purpose |
|--------|-----------------|---------|
| `search_offers()` | `POST /api/v0/bundles/` | Find GPU machines matching hardware requirements |
| `create_instance()` | `PUT /api/v0/asks/{offer_id}/` | Create instance from best offer |
| `get_instance()` | `GET /api/v0/instances/{id}/` | Poll instance status, get connection info |
| `stop_instance()` | `PUT /api/v0/instances/{id}/` | Stop instance (pause — retains disk, stops GPU billing) |
| `start_instance()` | `PUT /api/v0/instances/{id}/` | Restart stopped instance (resume) |
| `destroy_instance()` | `DELETE /api/v0/instances/{id}/` | Tear down instance permanently |

### 4.2 Offer Search → Instance Creation

When Apex provisions a node, it translates `HardwareRequirements` into Vast.ai search filters:

```python
# Vast.ai search payload derived from HardwareRequirements
{
    "gpu_name": {"in": hardware.gpu_whitelist},   # e.g. ["RTX_4090", "A100_SXM4"]
    "num_gpus": {"gte": hardware.num_gpus},
    "disk_space": {"gte": hardware.min_disk_gb},
    "inet_up": {"gte": hardware.min_network_upload_mbps},
    "inet_down": {"gte": hardware.min_network_download_mbps},
    "cuda_max_good": {"gte": float(hardware.cuda_min_version)},
    "verified": {"eq": True},
    "rentable": {"eq": True},
    "rented": {"eq": False},
    "type": "on-demand",
    "limit": 10,
    "order": "dph_total",           # cheapest first
    "order_dir": "asc"
}
```

**Offer selection strategy:** Pick the cheapest offer from the search results. If creation fails (offer taken between search and create), retry with the next offer. Up to `settings.max_node_provisioning_retries` total attempts across different offers.

### 4.3 Instance Creation Payload

```python
# PUT /api/v0/asks/{offer_id}/
{
    "image": "vastai/comfy:latest",   # default Docker image; custom image in later phases
    "disk": hardware.min_disk_gb,
    "runtype": "ssh_direct",
    "env": {
        # Aisha deployment
        "ACS_BUNDLE": bundle_name,
        "ACS_BUNDLE_VERSION": bundle_version or "current",
        "ACS_GITHUB_TOKEN": settings.ai_bundles_github_token,
        "ACS_BUNDLES_REPO": settings.ai_bundles_repo_url,
        "ACS_HF_TOKEN": settings.hf_token,
        "ACS_CIVITAI_API_TOKEN": settings.civitai_api_token,

        # Cloudflare Tunnel
        "ACS_CF_TUNNEL_TOKEN": tunnel_token,

        # Apex callback (Phase 2 — env vars set now, unused until Phase 2)
        "ACS_APEX_SESSION_ID": str(session_id),
        "ACS_APEX_CALLBACK_URL": settings.apex_callback_url,
        "ACS_APEX_CALLBACK_TOKEN": callback_token,

        # Port mapping for ComfyUI (internal, tunneled via CF)
        "-p 8188:8188": "1"
    },
    "onstart": "curl -sL https://raw.githubusercontent.com/gearbox/aisha/main/scripts/onstart.sh | bash"
}
```

---

## 5. Cloudflare Tunnel Integration

### 5.1 Tunnel Lifecycle

Each GPU session gets its own CF tunnel. The tunnel lifecycle is tied to the session lifecycle:

```
Session Created  →  Tunnel Created  →  DNS CNAME Created  →  Instance Created
                                                                    │
                                                              onstart.sh installs
                                                              cloudflared + connects
                                                                    │
Session Stopped  →  Instance Destroyed  →  Tunnel Deleted  →  DNS CNAME Deleted
```

For **paused** sessions, the tunnel and DNS record are retained. `cloudflared` on the node stops when the instance stops and automatically reconnects when the instance restarts.

### 5.2 Cloudflare Client

```
src/api/services/
└── cloudflare/
    ├── __init__.py
    ├── client.py         # CloudflareTunnelClient — create/delete tunnel, manage DNS + ingress
    ├── schemas.py        # msgspec Structs for CF API shapes
    └── exceptions.py     # CloudflareError
```

**Key operations:**

| Method | CF API Endpoint | Purpose |
|--------|----------------|---------|
| `create_tunnel()` | `POST /accounts/{id}/cfd_tunnel` | Create named tunnel, get token |
| `configure_tunnel()` | `PUT /accounts/{id}/cfd_tunnel/{tunnel_id}/configurations` | Set ingress rules (port 8188 → localhost) |
| `create_dns_record()` | `POST /zones/{id}/dns_records` | CNAME `gpu-{session_id}.gpu-domain.com` → `{tunnel_id}.cfargotunnel.com` |
| `delete_tunnel()` | `DELETE /accounts/{id}/cfd_tunnel/{tunnel_id}` | Cleanup on session stop (not pause) |
| `delete_dns_record()` | `DELETE /zones/{id}/dns_records/{record_id}` | Cleanup on session stop (not pause) |

### 5.3 Tunnel Configuration

When creating a tunnel via API with `config_src: "cloudflare"`, the ingress rules are managed remotely (no config file needed on the GPU node):

```python
# Tunnel ingress configuration (set via CF API)
{
    "config": {
        "ingress": [
            {
                "hostname": f"gpu-{short_session_id}.gpu-domain.com",
                "service": f"http://localhost:{comfyui_port}"
            },
            {
                "service": "http_status:404"  # catch-all required by CF
            }
        ]
    }
}
```

The GPU node runs: `cloudflared tunnel run --token $ACS_CF_TUNNEL_TOKEN` — zero local config.

### 5.4 Tunnel Hostname Convention

```
gpu-{short_session_id}.gpu-domain.com
```

Example: `gpu-01jf8x3k.gpu-domain.com` where `01jf8x3k` is the first 8 chars of the session UUIDv7.

> **Why short ID?** CF tunnel hostnames must be valid DNS labels. Full UUIDs work but are unwieldy in logs. The 8-char prefix of a UUIDv7 is timestamp-derived and collision-resistant for concurrent sessions.

> **Why prefix instead of subdomain?** Cloudflare Universal SSL (free plan) covers the zone apex (`gpu-domain.com`) and one wildcard level (`*.gpu-domain.com`). A nested subdomain like `*.gpu.gpu-domain.com` requires Advanced Certificate Manager. To keep infrastructure costs minimal, GPU session hostnames live one level below the apex with a `gpu-` prefix instead of a `gpu` subdomain.

### 5.5 Orphaned Tunnel Cleanup

A periodic cleanup task (runs hourly) reconciles CF tunnels against active sessions:

1. List all tunnels with name prefix `gpu-session-*` via CF API
2. For each tunnel, check if a corresponding `gpu_sessions` row exists with status in (`pending`, `provisioning`, `active`, `stale`, `stopping`, `paused`, `resuming`)
3. If no matching active session → delete tunnel + DNS record

This handles edge cases where Apex crashes after creating a tunnel but before recording it in the DB, or where session cleanup fails.

### 5.6 `onstart.sh` Changes

Add to the existing `onstart.sh`:

```bash
# ==============================================================================
# Cloudflare Tunnel (added for Apex integration)
# ==============================================================================
install_cloudflared() {
    if command -v cloudflared &>/dev/null; then
        log "cloudflared already installed"
        return 0
    fi
    log "Installing cloudflared..."
    curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
        -o /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
    log "cloudflared installed: $(cloudflared --version)"
}

start_tunnel() {
    local tunnel_token="${ACS_CF_TUNNEL_TOKEN:-}"
    if [[ -z "$tunnel_token" ]]; then
        log "No CF tunnel token provided, skipping tunnel setup"
        return 0
    fi
    log "Starting Cloudflare tunnel..."
    nohup cloudflared tunnel run --token "$tunnel_token" > /var/log/cloudflared.log 2>&1 &
    log "Cloudflare tunnel started (PID: $!)"
}

# Call early in the script (before bundle deployment starts)
install_cloudflared
start_tunnel
```

The tunnel starts early so Apex can begin probing even while the bundle is still deploying (probes will fail with connection refused until ComfyUI starts, which is expected and handled by the provisioning worker).

### 5.7 Instance Portal Named Tunnel

The Vast.ai template image ships an **Instance Portal** that runs a built-in `tunnel_manager`. By default it creates account-less quick tunnels (`*.trycloudflare.com`) for every service at boot. When the env var `CF_TUNNEL_TOKEN` is present at boot, the portal runs a **named tunnel** instead.

Apex injects `CF_TUNNEL_TOKEN` (set to the same value as `ACS_CF_TUNNEL_TOKEN`) into the per-instance env at creation time, so the portal sees it before the provisioning script runs. This is the **only mechanism** that starts the named tunnel — there is no separate `cloudflared` process managed by the Aisha script for named tunnels.

**Ingress is authoritative in Cloudflare.** Apex creates the tunnel with `config_src: "cloudflare"`, meaning the ingress rules are stored remotely in Cloudflare and pulled by the connector at startup. Apex sets the ingress via `configure_tunnel_ingress()` (pointing at `localhost:8188`). The portal connector fetches and honors that config — there is exactly one ingress config and no conflict.

**One token per instance.** Cloudflare enforces that each tunnel token can connect from exactly one location. Apex generates a fresh tunnel (and token) per GPU session. The token must not be shared across sessions.

---

## 6. R2 Content Sync

### 6.1 Output: ComfyUI → R2

When ComfyUI finishes a generation job, Apex retrieves the output and stores it in R2. The existing flow in `AishaJobService` remains unchanged except for the transport:

```
ComfyUI generates image/video
    → Apex polls job completion via tunnel
    → Apex downloads output: GET https://{tunnel_hostname}/view?filename=...
    → Apex uploads to R2: PUT users/{user_id}/outputs/{job_id}/{file_id}.{ext}
    → Apex creates GenerationOutput DB record
```

The tunnel provides encrypted HTTPS transport. Output files (images: 1-10MB, videos: 10-100MB) transfer efficiently through CF's network.

### 6.2 Input: R2 → ComfyUI (for i2i, i2v, flf2v)

For generation types that require input images/videos, the user's content is already stored in R2. ComfyUI needs it available locally. The approach:

**Apex bridges R2 to ComfyUI** — before submitting the workflow, Apex:

1. Downloads the input image from R2 (via `aioboto3`, fast — same Cloudflare network)
2. Uploads it to ComfyUI's input directory via `POST https://{tunnel_hostname}/upload/image`
3. References the uploaded filename in the workflow JSON

```python
# In AishaGenerationProvider.submit(), before workflow queue:
if request.input_image_id:
    # 1. Fetch from R2
    image_data = await r2_storage.get_object(
        key=f"users/{user_id}/uploads/{request.input_image_id}.{ext}"
    )
    # 2. Upload to ComfyUI via tunnel
    upload_result = await comfyui_client.upload_image(
        image_data=image_data,
        filename=f"input_{request.input_image_id}.{ext}",
    )
    # 3. Inject uploaded filename into workflow parameters
    input_filename = upload_result["name"]
```

This keeps ComfyUI completely unaware of R2. The bandwidth overhead is minimal (input images are typically < 10MB), and Apex ↔ R2 is fast since both use Cloudflare's network.

> **Future optimization (deferred):** A custom ComfyUI node that loads images directly from pre-signed R2 URLs would eliminate the double transfer. Not worth the complexity now.

---

## 7. Database Changes

### 7.1 `gpu_sessions` Table — Additional Columns

The existing `gpu_sessions` table needs new columns for provisioning, tunnel tracking, and pause/resume. Since we're pre-production, these go into the baseline migration:

```python
# New columns to add to GpuSession model

# Bundle identity
bundle_name: Mapped[str] = mapped_column(
    String(100), nullable=False,
    comment="ai-bundles bundle name (e.g. wan_2.2_i2v)",
)
bundle_version: Mapped[str | None] = mapped_column(
    String(20), nullable=True,
    comment="Bundle version (e.g. 260105-01). NULL = current",
)
model_type: Mapped[str] = mapped_column(
    String(50), nullable=False,
    comment="ModelType enum value this session serves",
)

# Cloudflare tunnel
cf_tunnel_id: Mapped[str | None] = mapped_column(
    String(64), nullable=True,
    comment="Cloudflare tunnel UUID",
)
cf_dns_record_id: Mapped[str | None] = mapped_column(
    String(64), nullable=True,
    comment="Cloudflare DNS record ID for cleanup",
)
tunnel_hostname: Mapped[str | None] = mapped_column(
    String(255), nullable=True,
    comment="Full tunnel hostname (e.g. gpu-01jf8x3k.gpu-domain.com)",
)

# Vast.ai instance details
vastai_offer_id: Mapped[int | None] = mapped_column(
    Integer, nullable=True,
    comment="Vast.ai offer ID used to create the instance",
)
vastai_cost_per_hour_micros: Mapped[int | None] = mapped_column(
    Integer, nullable=True,
    comment="Vast.ai $/hr in microdollars (1_000_000 = $1.00) at instance creation time",
)
vastai_gpu_name: Mapped[str | None] = mapped_column(
    String(50), nullable=True,
    comment="GPU model name from Vast.ai (e.g. RTX_4090)",
)

# Provisioning tracking
provision_attempt: Mapped[int] = mapped_column(
    Integer, nullable=False, server_default=text("1"),
    comment="Current provisioning attempt number (1-based)",
)

# Pause/resume tracking
paused_at: Mapped[datetime | None] = mapped_column(
    DateTime(timezone=True), nullable=True,
    comment="When the session was paused (Vast.ai instance stopped)",
)
resumed_at: Mapped[datetime | None] = mapped_column(
    DateTime(timezone=True), nullable=True,
    comment="When the last resume was requested",
)

# Phase 2 callback token (set at creation, unused until Phase 2)
callback_token: Mapped[str | None] = mapped_column(
    String(128), nullable=True,
    comment="HMAC token for Phase 2 node → Apex callback auth",
)
```

### 7.2 New Indexes

```python
# Unique constraint: one active session per ModelType per user per product
# "Active" = any non-terminal state (everything except stopped and failed)
Index(
    "ix_gpu_sessions_active_user_model",
    "user_id", "product_id", "model_type",
    unique=True,
    postgresql_where=text("status NOT IN ('stopped', 'failed')"),
)
```

This partial unique index enforces the "one active session per ModelType per user per product" constraint at the DB level. A paused session is still "active" in this sense — the user must stop or let it fail before starting a new session for the same model.

---

## 8. Session Lifecycle — State Machine

### 8.1 States

```python
class GpuSessionStatus(StrEnum):
    """GPU session lifecycle states."""

    pending = "pending"            # session requested, Vast.ai instance creating
    provisioning = "provisioning"  # Vast.ai instance running, waiting for ComfyUI
    active = "active"              # ComfyUI reachable, ready for generation jobs
    stale = "stale"                # was active, now unreachable (health reconciler)
    paused = "paused"              # user paused — Vast.ai instance stopped, disk retained
    resuming = "resuming"          # user resumed — Vast.ai instance restarting
    stopping = "stopping"          # user requested stop, teardown in progress
    stopped = "stopped"            # session ended normally, resources cleaned up
    failed = "failed"              # unrecoverable error
```

### 8.2 State Transitions

```
                    ┌────────────────────────────────────────────────────┐
                    │                                                    │
                    ▼                                                    │
┌─────────┐   ┌──────────────┐   ┌────────┐   ┌────────┐        ┌─────────┐
│ pending  │──>│ provisioning │──>│ active │──>│stopping│───────>│ stopped │
└─────────┘   └──────────────┘   └────────┘   └────────┘        └─────────┘
     │              │                │  ▲                               ▲
     │              │                │  │                               │
     │              ▼                ▼  │                               │
     │         ┌──────────┐    ┌────────┐                              │
     │         │  pending  │   │ stale  │──────── stopping ────────────┘
     │         │  (retry)  │   └────────┘
     │         └──────────┘      │  ▲
     │              │            ▼  │
     │              │       ┌────────────┐
     │              │       │  (recover) │ (health reconciler clears stale)
     │              │       └────────────┘
     │              │
     │              │        ┌────────┐    ┌──────────┐    ┌──────────┐
     │              │        │ active │──>│  paused  │──>│ resuming │──> active
     │              │        └────────┘    └──────────┘    └──────────┘
     │              │                           │                │
     │              │                      stopping          failed
     │              │                           │                │
     ▼              ▼                           ▼                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                              failed                                   │
└──────────────────────────────────────────────────────────────────────┘
```

### 8.3 Transition Table

| From | To | Trigger | Actions |
|------|----|---------|---------|
| `pending` | `provisioning` | Vast.ai instance status = `running` | Update session status |
| `pending` | `pending` (retry) | Vast.ai creation fails (offer taken) | Try next offer |
| `pending` | `failed` | Determined error (payment, no capacity after all retries) | Refund reservation, notify user |
| `provisioning` | `active` | ComfyUI `/object_info` returns 200 via tunnel | Set `started_at`, publish SSE |
| `provisioning` | `pending` (retry) | Timeout, attempt < `settings.max_node_provisioning_retries` | Destroy node, pick new offer |
| `provisioning` | `failed` | Timeout, attempt >= `settings.max_node_provisioning_retries` | Destroy node, delete tunnel, refund |
| `active` | `stale` | Health reconciler probe fails | Set `stale_detected_at` (existing mechanism) |
| `stale` | `active` | Health reconciler probe succeeds | Clear `stale_detected_at` (existing mechanism) |
| `active` | `paused` | User requests pause | Stop Vast.ai instance, set `paused_at` |
| `paused` | `resuming` | User requests resume | Start Vast.ai instance, set `resumed_at` |
| `resuming` | `active` | ComfyUI `/object_info` returns 200 via tunnel | Update `started_at` |
| `resuming` | `failed` | Vast.ai instance fails to restart | Destroy instance, delete tunnel |
| `active` / `stale` | `stopping` | User requests stop (confirmed) | Begin teardown |
| `paused` | `stopping` | User requests stop (confirmed) | Begin teardown |
| `stopping` | `stopped` | Instance destroyed, tunnel deleted | Set `stopped_at`, finalize billing |

### 8.4 Pause/Resume Details

**Pause** (`active` → `paused`):
1. Call Vast.ai `stop_instance()` — GPU stops, disk is retained
2. `cloudflared` process on the node terminates (instance is stopped)
3. CF tunnel stays configured (DNS record + tunnel config intact)
4. Session billing: GPU charges stop, only Vast.ai storage charges continue (minimal, ~$0.01-0.05/hr)
5. Set `paused_at = NOW()`, status = `paused`

**Resume** (`paused` → `resuming` → `active`):
1. Call Vast.ai `start_instance()` — instance restarts from disk image
2. `onstart.sh` runs again but is fast: repos are cached, models are on disk, `cloudflared` reconnects
3. Provisioning worker detects `resuming` session, probes ComfyUI via tunnel
4. Once ComfyUI responds: status → `active`, GPU billing resumes
5. Typical resume time: **10-30 seconds** (vs. 5-15 minutes for cold start)

---

## 9. Service Layer

### 9.1 Module Structure

```
src/api/services/
├── gpu_session/
│   ├── __init__.py
│   ├── service.py            # GpuSessionService — orchestrates the full lifecycle
│   ├── provisioning_worker.py # GpuProvisioningWorker — background polling loop
│   └── cleanup.py            # OrphanedTunnelCleanupWorker — periodic CF tunnel reconciliation
├── vastai/
│   ├── __init__.py
│   ├── client.py             # VastAIClient
│   ├── schemas.py            # Vast.ai API DTOs
│   └── exceptions.py
├── cloudflare/
│   ├── __init__.py
│   ├── client.py             # CloudflareTunnelClient
│   ├── schemas.py            # CF API DTOs
│   └── exceptions.py
└── bundle_index.py           # BundleIndexService — git sync + index parsing
```

### 9.2 `GpuSessionService`

The central orchestrator for session lifecycle:

```python
class GpuSessionService:
    """Manages the full GPU session lifecycle.

    Responsibilities:
    - Start: resolve bundle → create CF tunnel → search Vast.ai → create instance → persist session
    - Pause: stop Vast.ai instance, retain tunnel + disk
    - Resume: restart Vast.ai instance, wait for ComfyUI readiness
    - Stop: two-call confirmation → destroy instance → delete tunnel → update session
    - Status: query session state, return connection info
    - Routing: resolve active session for a given (user_id, product_id, model_type)
    """

    def __init__(
        self,
        vastai_client: VastAIClient,
        cf_client: CloudflareTunnelClient,
        bundle_index: BundleIndexService,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None: ...

    async def start_session(
        self,
        *,
        user_id: UUID,
        product_id: str,
        model_type: ModelType,
        bundle_override: str | None = None,  # admin only
        account_id: UUID,                     # for billing reservation
    ) -> GpuSession:
        """Create a new GPU session.

        Steps:
        1. Verify no active session exists for this (user, product, model_type)
        2. Resolve bundle (default or override)
        3. Load hardware requirements from bundle.yaml
        4. Reserve billing tokens (5-min minimum)
        5. Create CF tunnel + DNS record
        6. Search Vast.ai for matching offers
        7. Create Vast.ai instance with env vars
        8. Persist GpuSession with status='pending'
        9. Notify provisioning worker

        Raises:
            SessionAlreadyExistsError: active session exists for this model
            NoCapacityError: no Vast.ai offers match requirements
            InsufficientBalanceError: token balance too low for reservation
        """

    async def pause_session(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
    ) -> GpuSession:
        """Pause a GPU session (stop Vast.ai instance, retain disk).

        Only allowed from 'active' status.
        CF tunnel and DNS record are retained for fast resume.
        GPU billing stops; only Vast.ai storage charges continue.
        """

    async def resume_session(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
    ) -> GpuSession:
        """Resume a paused GPU session (restart Vast.ai instance).

        Only allowed from 'paused' status.
        Transitions to 'resuming'. Provisioning worker monitors
        until ComfyUI becomes reachable, then transitions to 'active'.
        """

    async def stop_session(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
        confirmed: bool = False,
    ) -> GpuSession | StopConfirmation:
        """Stop a GPU session permanently (two-call confirmation flow).

        Allowed from 'active', 'stale', or 'paused' status.
        First call (confirmed=False): returns StopConfirmation with cost summary.
        Second call (confirmed=True): destroys instance, deletes tunnel, updates status.

        Teardown order:
        1. Update session status → 'stopping'
        2. Destroy Vast.ai instance (or skip if already stopped/paused)
        3. Delete CF tunnel + DNS record
        4. Update session status → 'stopped', set stopped_at
        5. Process billing (finalize charges, refund unused reservation if applicable)
        """

    async def get_session(
        self,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
    ) -> GpuSession | None:
        """Get session by ID with ownership check."""

    async def get_active_session_for_model(
        self,
        user_id: UUID,
        product_id: str,
        model_type: ModelType,
    ) -> GpuSession | None:
        """Resolve the active session for generation routing.

        Only returns sessions with status='active'.
        Paused/stale sessions are not routable.
        """

    async def list_user_sessions(
        self,
        user_id: UUID,
        product_id: str,
        include_stopped: bool = False,
    ) -> Sequence[GpuSession]:
        """List user's sessions, optionally including terminal states."""
```

### 9.3 `GpuProvisioningWorker`

Background worker that monitors `pending`, `provisioning`, and `resuming` sessions:

```python
class GpuProvisioningWorker:
    """Polls pending/provisioning/resuming GPU sessions until ready or failed.

    Lifecycle: started in app lifespan, runs as asyncio.Task.

    For each non-terminal session:
    1. If status='pending' and vastai_instance_id is set:
       - Poll Vast.ai instance status
       - If 'running': transition to 'provisioning', start probing ComfyUI
    2. If status='provisioning':
       - Probe ComfyUI via tunnel: GET https://{tunnel_hostname}/object_info
       - If 200 OK: transition to 'active', set started_at
       - If timeout exceeded: mark as failed or retry with new node
    3. If status='resuming':
       - Same probing logic as 'provisioning' but with shorter timeout
       - Resume typically completes in 10-30 seconds

    Retry logic (provisioning only, not resume):
    - On provisioning failure (node unreachable after max timeout):
      1. Destroy failed Vast.ai instance
      2. Increment provision_attempt
      3. If attempt < settings.max_node_provisioning_retries:
         search new offer → create new instance → stay in 'pending'
      4. If attempt >= settings.max_node_provisioning_retries:
         transition to 'failed', delete tunnel, refund billing reservation
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        vastai_client: VastAIClient,
        http_client: httpx.AsyncClient,
        settings: Settings,
    ) -> None: ...

    # Poll interval: settings.gpu_provision_poll_interval_seconds (default 15)
    # Max provisioning timeout: settings.gpu_provision_timeout_minutes (default 20) per attempt
    # Max resume timeout: settings.gpu_resume_timeout_minutes (default 5)
```

### 9.4 Generation Routing

The existing `AishaGenerationProvider` needs to route to the user's active session instead of a fixed `comfyui_host:comfyui_port`:

```python
class AishaGenerationProvider:
    """Adapts ComfyUI to the GenerationProvider protocol.

    CHANGED: Instead of a single ComfyUIClient, resolves the user's
    active session and creates a session-specific client.
    """

    def __init__(
        self,
        workflow_service: WorkflowService,
        gpu_session_service: GpuSessionService,
    ) -> None:
        self._workflow = workflow_service
        self._gpu_session_service = gpu_session_service

    async def submit(self, request, *, user_id, session, ...) -> GenerationJob:
        # 1. Resolve active session for this model type
        gpu_session = await self._gpu_session_service.get_active_session_for_model(
            user_id=user_id,
            product_id=product_id,
            model_type=request.model,
        )
        if gpu_session is None:
            raise NoActiveSessionError(
                f"No active GPU session for {request.model.value}. "
                "Start a session first via POST /v1/sessions."
            )

        # 2. Create session-specific ComfyUI client
        comfyui_client = ComfyUIClient(
            host=gpu_session.tunnel_hostname,
            port=443,      # HTTPS via CF tunnel
            scheme="https",
        )

        # 3. Bridge input image from R2 to ComfyUI if needed
        if request.input_image_id:
            input_filename = await self._upload_input_from_r2(
                comfyui_client, request.input_image_id, user_id
            )
            # inject into workflow params

        # 4. Submit workflow via this client
        # ... rest of existing logic using comfyui_client
```

---

## 10. API Endpoints

### 10.1 User-Facing Session Endpoints

```
POST   /v1/sessions                         # Start a new GPU session
GET    /v1/sessions                         # List user's sessions
GET    /v1/sessions/{session_id}            # Get session status + details
POST   /v1/sessions/{session_id}/stop       # Stop session (two-call confirmation)
POST   /v1/sessions/{session_id}/pause      # Pause session (stop GPU, retain disk)
POST   /v1/sessions/{session_id}/resume     # Resume paused session
```

### 10.2 Request/Response Schemas

```python
# POST /v1/sessions — Request
class CreateSessionRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    model: ModelType
    """Which model to provision (e.g. aisha-video)."""

    bundle_override: str | None = None
    """Admin only: specific bundle name or name:version. Ignored for non-admin users."""


# POST /v1/sessions — Response (202 Accepted)
class SessionResponse(msgspec.Struct, kw_only=True):
    id: UUID
    status: GpuSessionStatus
    model_type: str
    bundle_name: str
    gpu_name: str | None           # Set after Vast.ai offer selected
    tunnel_hostname: str | None    # Set after CF tunnel created
    created_at: datetime
    started_at: datetime | None    # Set when ComfyUI becomes reachable
    paused_at: datetime | None     # Set when user pauses
    stopped_at: datetime | None
    error_message: str | None
    provision_attempt: int


# POST /v1/sessions/{id}/stop (confirmed=false) — Response
class StopConfirmation(msgspec.Struct, kw_only=True):
    session_id: UUID
    active_duration_minutes: float
    message: str                   # Human-readable cost summary
    confirmed: bool = False        # Client must re-send with confirmed=true


# GET /v1/sessions — Response
class SessionListResponse(msgspec.Struct, kw_only=True):
    sessions: list[SessionResponse]
```

### 10.3 Admin Endpoints

```
GET    /v1/admin/sessions              # List all sessions (any user, any product)
GET    /v1/admin/sessions/{id}         # Get session details (with Vast.ai metadata)
DELETE /v1/admin/sessions/{id}         # Force-terminate session (skip confirmation)
```

### 10.4 SSE Events

Session status changes are published via the existing `EventBus` (Redis Pub/Sub):

```python
# Channel: gpu_session:{user_id}:{product_id}
{
    "event": "session.status_changed",
    "data": {
        "session_id": "...",
        "status": "active",
        "model_type": "aisha-video",
        "tunnel_hostname": "gpu-01jf8x3k.gpu-domain.com"
    }
}
```

Frontend subscribes via existing SSE infrastructure to show real-time provisioning progress.

---

## 11. Configuration — New Settings

```python
# Added to src/core/config.py → Settings

# --- GPU Session Provisioning ---
vastai_api_key: str = Field(default="", description="Vast.ai API key")
ai_bundles_github_token: str = Field(default="", description="GitHub PAT for ai-bundles private repo")
ai_bundles_repo_url: str = Field(
    default="https://github.com/gearbox/ai-bundles.git",
    description="Git URL for the ai-bundles repository",
)
ai_bundles_sync_interval_minutes: int = Field(
    default=15, ge=1, le=60,
    description="How often to git-pull ai-bundles for updates",
)
hf_token: str = Field(default="", description="HuggingFace token for private model downloads")
civitai_api_token: str = Field(default="", description="Civitai API token for model downloads")

# --- Cloudflare API (per-session GPU node tunnels — NOT the apex API's own cloudflared sidecar) ---
aisha_cf_api_token: str = Field(default="", description="Cloudflare API token (Tunnel:Edit permission) for per-session GPU node tunnels")
aisha_cf_account_id: str = Field(default="", description="Cloudflare account ID for per-session GPU node tunnels")
aisha_cf_zone_id: str = Field(default="", description="Cloudflare zone ID for the GPU tunnel domain")
aisha_cf_tunnel_domain: str = Field(
    default="gpu-domain.com",
    description="CF zone for tunnel DNS records. Tunnel hostnames are constructed as gpu-{session_id_short}.{this}.",
)

# --- Provisioning Worker ---
gpu_provision_poll_interval_seconds: int = Field(
    default=15, ge=5, le=60,
    description="How often to poll pending/provisioning/resuming sessions",
)
gpu_provision_timeout_minutes: int = Field(
    default=20, ge=5, le=60,
    description="Max time to wait for a single provisioning attempt",
)
gpu_resume_timeout_minutes: int = Field(
    default=5, ge=1, le=15,
    description="Max time to wait for a paused session to resume",
)
max_node_provisioning_retries: int = Field(
    default=3, ge=1, le=10,
    description="Max provisioning attempts before marking session as failed",
)

# --- Phase 2 Callback (pre-wired) ---
apex_callback_url: str = Field(
    default="",
    description="Public URL for GPU node → Apex callbacks (Phase 2)",
)
```

---

## 12. Environment Variables Passed to Vast.ai Instance

All env vars set at instance creation, covering both Phase 1 and Phase 2:

| Env Var | Source | Phase | Purpose |
|---------|--------|-------|---------|
| `ACS_BUNDLE` | Bundle index resolution | 1 | Bundle to deploy |
| `ACS_BUNDLE_VERSION` | Bundle index or override | 1 | Specific version or "current" |
| `ACS_GITHUB_TOKEN` | `settings.ai_bundles_github_token` | 1 | Clone ai-bundles (private repo) |
| `ACS_BUNDLES_REPO` | `settings.ai_bundles_repo_url` | 1 | Git URL for ai-bundles |
| `ACS_HF_TOKEN` | `settings.hf_token` | 1 | HuggingFace model downloads |
| `ACS_CIVITAI_API_TOKEN` | `settings.civitai_api_token` | 1 | Civitai model downloads |
| `ACS_CF_TUNNEL_TOKEN` | CF API response | 1 | Cloudflare tunnel auth token |
| `ACS_APEX_SESSION_ID` | `str(session.id)` | 2 | Session ID for callback correlation |
| `ACS_APEX_CALLBACK_URL` | `settings.apex_callback_url` | 2 | Apex webhook URL |
| `ACS_APEX_CALLBACK_TOKEN` | Generated HMAC token | 2 | Callback authentication |

---

## 13. Error Handling & Retry Strategy

### 13.1 Determined Failures (No Retry)

These are unrecoverable — transition immediately to `failed`, notify user, refund reservation:

| Error | Detection | User Message |
|-------|-----------|-------------|
| Vast.ai payment issue | HTTP 402 / insufficient credits response | "GPU provider billing issue. Please contact support." |
| No matching offers | Empty search results | "No GPU capacity available for this model. Try again later." |
| CF tunnel creation fails | CF API error | "Infrastructure setup failed. Please try again." |
| User balance too low | `InsufficientBalanceError` from billing | "Insufficient token balance for GPU session." |

### 13.2 Transient Failures (Retry with New Node)

These trigger a retry cycle — destroy current node, pick a new offer, re-attempt:

| Error | Detection | Max Retries |
|-------|-----------|-------------|
| Node unreachable after boot | Vast.ai shows `running` but ComfyUI never responds via tunnel | `settings.max_node_provisioning_retries` |
| `onstart.sh` failure | Node process exits / ComfyUI never starts within timeout | `settings.max_node_provisioning_retries` |
| Vast.ai instance creation fails (offer taken) | HTTP 409 or similar | Try next offer from search results |

**Retry flow:**
1. Destroy the failed Vast.ai instance (best effort — if destroy fails, it'll be caught by billing reconciliation)
2. CF tunnel is reused (same tunnel, same hostname — just the instance changes)
3. Increment `provision_attempt` on the session
4. Search for a new offer (exclude the failed machine if possible)
5. Create new instance with same env vars
6. If `provision_attempt > settings.max_node_provisioning_retries`: transition to `failed`, delete tunnel, refund

### 13.3 Cleanup on Failure

When a session reaches `failed` status:
1. Destroy Vast.ai instance (if exists)
2. Delete CF tunnel + DNS record (if exists)
3. Refund billing reservation (if reserved)
4. Publish SSE event `session.failed`

---

## 14. Observability

### 14.1 Structured Log Events

```
gpu_session.create_requested    — user_id, product_id, model_type
gpu_session.bundle_resolved     — bundle_name, bundle_version, hardware_reqs
gpu_session.tunnel_created      — session_id, tunnel_id, hostname
gpu_session.offer_selected      — session_id, offer_id, gpu_name, dph_total
gpu_session.instance_created    — session_id, vastai_instance_id
gpu_session.instance_running    — session_id (Vast.ai reports running)
gpu_session.comfyui_probe       — session_id, hostname, status_code, latency_ms
gpu_session.active              — session_id (ComfyUI reachable, session ready)
gpu_session.provision_timeout   — session_id, attempt, elapsed_minutes
gpu_session.retry               — session_id, attempt, reason
gpu_session.failed              — session_id, reason, attempts_exhausted
gpu_session.pause_requested     — session_id
gpu_session.paused              — session_id
gpu_session.resume_requested    — session_id
gpu_session.resumed             — session_id
gpu_session.stop_requested      — session_id, confirmed
gpu_session.instance_destroyed  — session_id, vastai_instance_id
gpu_session.tunnel_deleted      — session_id, tunnel_id
gpu_session.stopped             — session_id, total_duration_minutes
gpu_session.orphan_cleanup      — tunnel_id, reason
gpu_session.r2_input_bridged    — session_id, image_id, size_bytes
```

### 14.2 Health Check Integration

The existing `GpuSessionReconciler` already probes active sessions. With CF tunnels, it probes via `https://{tunnel_hostname}/object_info` instead of `http://{node_host}:{node_port}/object_info`. The reconciler needs a minor update to support HTTPS tunnel hostnames alongside direct IP:port (for backward compatibility during migration).

Paused and resuming sessions are excluded from reconciler probing (they are expected to be unreachable).

---

## 15. Implementation Phases

### Phase 1A: Foundation (No Vast.ai calls yet)
- `bundle_config.py` — `HardwareRequirements`, `BundleMapping` dataclasses
- `BundleIndexService` — git sync + index parsing
- `GpuSessionStatus` enum additions (`paused`, `resuming`)
- `gpu_sessions` table additions — new columns in squashed migration
- `GpuSessionRepository` — CRUD + unique constraint queries
- Settings additions

### Phase 1B: Cloudflare Tunnel Client
- `CloudflareTunnelClient` — create/configure/delete tunnel + DNS
- Unit tests with mocked CF API responses

### Phase 1C: Vast.ai Client
- `VastAIClient` — search offers, create/get/stop/start/destroy instance
- Offer → search filter translation from `HardwareRequirements`
- Unit tests with mocked Vast.ai API responses

### Phase 1D: Session Service + Provisioning Worker
- `GpuSessionService` — full lifecycle orchestration (start, pause, resume, stop)
- `GpuProvisioningWorker` — polling loop for pending/provisioning/resuming sessions
- Retry logic and error classification
- `OrphanedTunnelCleanupWorker`
- Integration tests

### Phase 1E: API Routes + Generation Routing
- Session CRUD endpoints (`/v1/sessions/*`)
- Admin session endpoints (`/v1/admin/sessions/*`)
- `AishaGenerationProvider` refactor for dynamic routing + R2 input bridging
- SSE events for session status changes
- `onstart.sh` changes for `cloudflared`

### Phase 1F: End-to-End Testing + `onstart.sh` Validation
- Full provisioning flow test (with real or mocked Vast.ai sandbox)
- Tunnel connectivity verification
- Generation job routing through tunnel
- R2 input/output sync verification
- Pause/resume cycle testing
- Failure/retry scenario testing

### Phase 2 (Future): Callback-Based Readiness
- Apex endpoint: `POST /v1/internal/sessions/{id}/ready` (authenticated via `ACS_APEX_CALLBACK_TOKEN`)
- Aisha `onstart.sh` addition: call callback after bundle deployment success
- Provisioning worker falls back to polling if callback doesn't arrive within timeout

---

## 16. Security Considerations

| Concern | Mitigation |
|---------|-----------|
| ComfyUI exposed to internet | CF Tunnel — no public ports, outbound-only connection from GPU node |
| Tunnel token in Vast.ai env vars | Vast.ai env vars are only visible to the instance owner; tunnel is scoped to a single session and deleted on cleanup |
| Phase 2 callback auth | HMAC token generated per-session, validated by Apex; prevents spoofed readiness signals |
| GitHub PAT in env vars | Read-only PAT scoped to `ai-bundles` repo only; rotatable |
| Admin bundle override | Guarded by `admin_guard` — only admin users can specify `bundle_override` |
| Orphaned resources | Cleanup workers for both CF tunnels and Vast.ai instances; billing reconciler catches billing drift |

---

## 17. Open Questions / Future Considerations

1. **Session warm pool**: Pre-provision a pool of stopped Vast.ai instances with common bundles. GPU charges stop when instances are stopped; only storage charges continue (cheap). Restart takes seconds vs. minutes for cold start. Deferred — revisit after launch.

2. **Custom Docker image**: Start with `vastai/comfy:latest` (pre-installed ComfyUI). In later phases, create a custom image with `cloudflared` pre-baked to save ~10s per provisioning.

3. **Multi-region**: Cheapest node strategy (no geo preference). Image/video generation is node-local; only results transfer to R2. Network speed filters in hardware requirements ensure adequate bandwidth.
