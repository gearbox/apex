# Web Push Setup Guide

How to enable Web Push notifications on an Apex deployment. For the API
contract (endpoints, wire payload, event → notification mapping), see
`docs/BACKEND_API_REFERENCE.md` §15b.

## How it's gated

Push is **off by default**. It turns on only when all four of these are set:

```bash
VAPID_PUBLIC_KEY=...
VAPID_PRIVATE_KEY=...
VAPID_SUBJECT=mailto:ops@apex.ai
REDIS_URL=redis://...
```

This is `Settings.push_enabled` (`src/core/config.py`). When any one is
missing, `GET/POST/DELETE /v1/push/*` return `503 Service Unavailable` and the
`push_dispatcher` worker never starts — no partial/half-configured state.

## Step 1 — Generate a VAPID key pair

One key pair per environment (dev, staging, prod). Never reuse a prod key pair
in dev, and never commit the private key anywhere.

```bash
uv run python tools/generate_vapid_keys.py
```

Output:

```
VAPID_PUBLIC_KEY=BN8x...
VAPID_PRIVATE_KEY=Yt3z...
VAPID_SUBJECT=mailto:ops@apex.ai  # replace with a real contact
```

`VAPID_SUBJECT` is the `sub` claim push services (FCM, Mozilla autopush, ...)
use to contact you if your server misbehaves — set it to a real mailbox or
`https://` URL you monitor, not the placeholder above.

## Step 2 — Set the environment variables

Add the three lines from Step 1 plus `REDIS_URL` to your `.env` (or
docker-compose env / secrets manager for staging/prod):

```bash
VAPID_PUBLIC_KEY=BN8x...
VAPID_PRIVATE_KEY=Yt3z...
VAPID_SUBJECT=mailto:you@yourdomain.com
REDIS_URL=redis://redis:6379/0   # already required for SSE — see §15
```

Optional tuning:

```bash
PUSH_BROADCAST_CONCURRENCY=10   # max concurrent sends per broadcast batch (default 10)
```

## Step 3 — Run the migration

```bash
make migrate
```

Adds the `push_subscriptions` table (`alembic/versions/015_add_push_subscriptions.py`).

## Step 4 — Restart and verify

```bash
make down && make dev
```

Confirm the dispatcher started:

```bash
docker compose logs api | grep push
# push_service.initialized
# push_dispatcher.started
```

Confirm the API surface is live (needs a valid JWT — see §2 of the API
reference for how to obtain one):

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/v1/push/vapid-public-key
# {"public_key":"BN8x..."}
```

If you get `503` instead, one of the four settings above isn't picked up —
double check the container actually has them (`docker compose exec api env | grep VAPID`).

## Step 5 — Frontend integration

The frontend PWA (separate repo) needs to:

1. `GET /v1/push/vapid-public-key` → pass `public_key` as `applicationServerKey`
   to `registration.pushManager.subscribe()`.
2. `POST /v1/push/subscriptions` with the resulting `PushSubscription.toJSON()`.
3. Handle the `push` event in its service worker and call
   `registration.showNotification(payload.title, ...)` using the wire payload
   fields (`title`, `body`, `url`, `tag`, `category`, `level`).
4. `DELETE /v1/push/subscriptions` on logout or when the user revokes
   notification permission.

Full request/response shapes, the wire payload contract, and a working
frontend code sample are in `docs/BACKEND_API_REFERENCE.md` §15b.

## Rotating or revoking keys

Changing `VAPID_PRIVATE_KEY`/`VAPID_PUBLIC_KEY` invalidates every existing
subscription (browsers pin a subscription to the public key used at
`subscribe()` time). After a rotation:

- Existing subscriptions will fail with `410 Gone` on next push attempt —
  `PushService` prunes them from `push_subscriptions` automatically, no
  manual cleanup needed.
- Every client must re-subscribe (re-run Step 5) to receive push again.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| All three endpoints return `503` | One of `VAPID_PUBLIC_KEY`/`VAPID_PRIVATE_KEY`/`VAPID_SUBJECT`/`REDIS_URL` is unset — check `push_enabled` inputs exactly |
| Endpoints work, but no notifications ever arrive | `push_dispatcher.started` missing from logs — check `WORKER_MODE` isn't `api_only` on this process |
| Notifications stop for one user only | Their subscription likely expired (410) and was pruned — have them reopen the app to re-subscribe |
| `pywebpush` raises on send (5xx from push service) | Transient — logged and skipped per-subscription, no retry in v1 (see §15b "Delivery Guarantees") |
