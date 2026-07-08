"""Apex background workers.

All periodic workers subclass ``PeriodicWorker`` (``src.workers.base``), which
owns the start/stop lifecycle, an optional Redis-backed leader lease (so only
one process ticks in multi-worker Granian deployments), jittered sleeps, and
a liveness heartbeat consumed by ``WorkerHeartbeatChecker``
(``src.api.services.health.checkers.workers``).

Whether a given process starts any workers at all is controlled by
``Settings.worker_mode`` (env ``WORKER_MODE``): ``all`` (default) starts every
worker alongside the API; ``api_only`` starts none.

In-process workers (started from the app lifespan, see
``src.api.dependencies.common.init_services``):
- ``token_cleanup`` (``src.workers.token_cleanup.TokenCleanupWorker``)
- ``aisha_job_poller`` (``src.workers.aisha_job_poller.AishaJobPoller``)
- ``gpu_provisioning`` (``src.api.services.gpu_session.provisioning_worker.GpuProvisioningWorker``)
- ``gpu_orphan_cleanup`` (``src.api.services.gpu_session.cleanup_worker.OrphanedTunnelCleanupWorker``)
- ``billing_reconciler`` (``src.api.services.gpu_session.billing_reconciler_worker.BillingReconcilerWorker``)
- ``health_snapshot`` / ``health_snapshot_cleanup`` (``src.api.services.health.worker``)
- ``grok_video_poller`` (``src.api.services.grok.video_worker.GrokVideoWorker``) — also
  runnable as a standalone process:

    python -m src.workers.grok_video
"""

from __future__ import annotations
