"""In-memory job manager for tracking generation jobs."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.api.schemas.generation import GenerationRequest, JobStatus
from src.api.services.comfyui_client import ComfyUIClient

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """Internal job representation."""

    job_id: str
    prompt_id: str | None  # ComfyUI prompt ID
    name: str
    request: GenerationRequest
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: float = 0.0
    images: list[str] = field(default_factory=list)
    error: str | None = None

    # ComfyUI-specific tracking
    input_images: list[str] = field(default_factory=list)


class JobManager:
    """Manages generation jobs lifecycle and status tracking.

    Provides:
    - Job creation and tracking
    - Status polling and updates
    - Image URL retrieval from ComfyUI history
    """

    def __init__(self, comfyui_client: ComfyUIClient) -> None:
        """Initialize job manager.

        Args:
            comfyui_client: ComfyUI client for API communication.
        """
        self._client = comfyui_client
        self._jobs: dict[str, Job] = {}
        self._prompt_to_job: dict[str, str] = {}  # prompt_id -> job_id mapping

    def create_job(self, request: GenerationRequest) -> Job:
        """Create a new job entry.

        Args:
            request: Generation request parameters.

        Returns:
            Created job instance.
        """
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            prompt_id=None,
            name=request.name or "Untitled",
            request=request,
            status=JobStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )
        self._jobs[job_id] = job
        logger.info(f"Created job {job_id}: {job.name}")
        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID.

        Args:
            job_id: Job identifier.

        Returns:
            Job if found, None otherwise.
        """
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 100,
    ) -> list[Job]:
        """List jobs with optional filtering.

        Args:
            status: Filter by status (optional).
            limit: Maximum jobs to return.

        Returns:
            List of jobs matching criteria.
        """
        jobs = list(self._jobs.values())

        if status is not None:
            jobs = [j for j in jobs if j.status == status]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def set_queued(self, job_id: str, prompt_id: str) -> None:
        """Mark job as queued with ComfyUI prompt ID.

        Args:
            job_id: Job identifier.
            prompt_id: ComfyUI prompt ID.
        """
        if job := self._jobs.get(job_id):
            job.prompt_id = prompt_id
            job.status = JobStatus.QUEUED
            job.started_at = datetime.now(timezone.utc)
            self._prompt_to_job[prompt_id] = job_id
            logger.debug(f"Job {job_id} queued with prompt_id {prompt_id}")

    def set_running(self, job_id: str, progress: float = 0.0) -> None:
        """Mark job as running with progress.

        Args:
            job_id: Job identifier.
            progress: Progress percentage (0-100).
        """
        if job := self._jobs.get(job_id):
            job.status = JobStatus.RUNNING
            job.progress = progress
            if job.started_at is None:
                job.started_at = datetime.now(timezone.utc)

    def set_completed(self, job_id: str, images: list[str]) -> None:
        """Mark job as completed with result images.

        Args:
            job_id: Job identifier.
            images: List of image URLs.
        """
        if job := self._jobs.get(job_id):
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.completed_at = datetime.now(timezone.utc)
            job.images = images
            logger.info(f"Job {job_id} completed with {len(images)} images")

    def set_failed(self, job_id: str, error: str) -> None:
        """Mark job as failed with error message.

        Args:
            job_id: Job identifier.
            error: Error description.
        """
        if job := self._jobs.get(job_id):
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error = error
            logger.error(f"Job {job_id} failed: {error}")

    async def poll_job_status(self, job_id: str) -> Job | None:
        """Poll ComfyUI for job status updates.

        Checks ComfyUI history and queue to determine current job state.

        Args:
            job_id: Job identifier.

        Returns:
            Updated job, or None if not found.
        """
        job = self._jobs.get(job_id)
        if not job or not job.prompt_id:
            return job

        # Skip if already terminal
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            return job

        try:
            # Check history for completion
            history = await self._client.get_history(job.prompt_id)

            if job.prompt_id in history:
                # Job has history entry - completed or failed
                prompt_history = history[job.prompt_id]

                if "outputs" in prompt_history:
                    # Extract images from outputs
                    images = self._extract_images_from_history(prompt_history)
                    self.set_completed(job_id, images)
                elif "status" in prompt_history:
                    status_info = prompt_history["status"]
                    if status_info.get("status_str") == "error":
                        error_msg = status_info.get("messages", [["Error", "Unknown error"]])
                        self.set_failed(job_id, str(error_msg))
            else:
                # Not in history - check queue
                queue = await self._client.get_queue()

                running = queue.get("queue_running", [])
                pending = queue.get("queue_pending", [])

                # Check if in running queue
                is_running = any(item[1] == job.prompt_id for item in running if len(item) > 1)

                if is_running:
                    self.set_running(job_id, 50.0)  # Approximate progress
                else:
                    # Check if still pending
                    is_pending = any(item[1] == job.prompt_id for item in pending if len(item) > 1)
                    if is_pending:
                        job.status = JobStatus.QUEUED

        except Exception as e:
            logger.warning(f"Error polling job {job_id}: {e}")

        return job

    def _extract_images_from_history(self, history: dict[str, Any]) -> list[str]:
        """Extract image URLs from ComfyUI history output.

        Args:
            history: History entry for a prompt.

        Returns:
            List of image URLs.
        """
        images: list[str] = []
        outputs = history.get("outputs", {})

        for _node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    filename = img_info.get("filename", "")
                    subfolder = img_info.get("subfolder", "")
                    img_type = img_info.get("type", "output")

                    if filename:
                        url = self._client.get_image_url(
                            filename=filename,
                            subfolder=subfolder,
                            folder_type=img_type,
                        )
                        images.append(url)

        return images

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Remove jobs older than specified age.

        Args:
            max_age_hours: Maximum job age in hours.

        Returns:
            Number of jobs removed.
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        old_jobs = [job_id for job_id, job in self._jobs.items() if job.created_at < cutoff]

        for job_id in old_jobs:
            job = self._jobs.pop(job_id, None)
            if job and job.prompt_id:
                self._prompt_to_job.pop(job.prompt_id, None)

        if old_jobs:
            logger.info(f"Cleaned up {len(old_jobs)} old jobs")

        return len(old_jobs)
