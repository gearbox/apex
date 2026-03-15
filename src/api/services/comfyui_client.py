"""ComfyUI HTTP client service for API communication."""

from typing import Any

import httpx
import structlog

from src.core.config import Settings

logger = structlog.get_logger(__name__)


class ComfyUIClientError(Exception):
    """Base exception for ComfyUI client errors."""

    pass


class ComfyUIConnectionError(ComfyUIClientError):
    """Raised when connection to ComfyUI fails."""

    pass


class ComfyUIAPIError(ComfyUIClientError):
    """Raised when ComfyUI API returns an error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ComfyUIClient:
    """Async HTTP client for ComfyUI API.

    Handles all communication with ComfyUI server including:
    - Workflow prompts submission
    - Image uploads
    - History/status queries
    - Image retrieval
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.comfyui_base_url
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ComfyUIClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Initialize HTTP client connection."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
            logger.info("comfyui.connected", url=self._base_url)

    async def close(self) -> None:
        """Close HTTP client connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("comfyui.disconnected")

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client, raising if not connected."""
        if self._client is None:
            raise ComfyUIConnectionError("Client not connected. Call connect() first.")
        return self._client

    async def health_check(self) -> bool:
        """Check if ComfyUI server is reachable.

        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            response = await self.client.get("/system_stats")
            return response.status_code == 200
        except httpx.RequestError as e:
            logger.warning("comfyui.health_check_failed", error=str(e))
            return False

    async def queue_prompt(self, workflow: dict[str, Any]) -> dict[str, Any]:
        """Submit a workflow prompt to ComfyUI queue.

        Args:
            workflow: Complete workflow dictionary with node configurations.

        Returns:
            Response containing prompt_id and other queue info.

        Raises:
            ComfyUIAPIError: If the API returns an error.
            ComfyUIConnectionError: If connection fails.
        """
        try:
            response = await self.client.post(
                "/prompt",
                json={"prompt": workflow},
            )

            if response.status_code != 200:
                error_text = response.text
                logger.error(
                    "comfyui.queue_prompt_failed",
                    status_code=response.status_code,
                    error=error_text,
                )
                raise ComfyUIAPIError(
                    f"Failed to queue prompt: {error_text}",
                    status_code=response.status_code,
                )

            result: dict[str, Any] = response.json()
            logger.debug("comfyui.prompt_queued", prompt_id=result.get("prompt_id"))
            return result

        except httpx.RequestError as e:
            logger.error("comfyui.connection_error", error=str(e))
            raise ComfyUIConnectionError(f"Failed to connect to ComfyUI: {e}") from e

    async def get_history(self, prompt_id: str) -> dict[str, Any]:
        """Get execution history for a prompt.

        Args:
            prompt_id: The prompt ID to query.

        Returns:
            History data for the prompt, empty dict if not found.
        """
        try:
            response = await self.client.get(f"/history/{prompt_id}")

            data: dict[str, Any] = response.json() if response.status_code == 200 else {}
            return data
        except httpx.RequestError as e:
            logger.warning("comfyui.get_history_failed", prompt_id=prompt_id, error=str(e))
            return {}

    async def get_queue(self) -> dict[str, Any]:
        """Get current ComfyUI queue status.

        Returns:
            Queue info with running and pending items.
        """
        try:
            response = await self.client.get("/queue")
            if response.status_code == 200:
                queue_data: dict[str, Any] = response.json()
                return queue_data
            return {"queue_running": [], "queue_pending": []}
        except httpx.RequestError as e:
            logger.warning("comfyui.get_queue_failed", error=str(e))
            return {"queue_running": [], "queue_pending": []}

    async def upload_image(
        self,
        image_data: bytes,
        filename: str,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        """Upload an image to ComfyUI input folder.

        Args:
            image_data: Raw image bytes.
            filename: Filename for the uploaded image.
            overwrite: Whether to overwrite existing file.

        Returns:
            Upload result with name, subfolder, and type.

        Raises:
            ComfyUIAPIError: If upload fails.
        """
        try:
            files = {"image": (filename, image_data)}
            data = {"overwrite": str(overwrite).lower()}

            response = await self.client.post(
                "/upload/image",
                files=files,
                data=data,
            )

            if response.status_code != 200:
                raise ComfyUIAPIError(
                    f"Image upload failed: {response.text}",
                    status_code=response.status_code,
                )

            upload_result: dict[str, Any] = response.json()
            logger.debug("comfyui.image_uploaded", name=upload_result.get("name"))
            return upload_result

        except httpx.RequestError as e:
            logger.error("comfyui.image_upload_failed", error=str(e))
            raise ComfyUIConnectionError(f"Failed to upload image: {e}") from e

    def get_image_url(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> str:
        """Construct URL for retrieving an image from ComfyUI.

        Args:
            filename: Image filename.
            subfolder: Subfolder within the type folder.
            folder_type: Folder type (output, input, temp).

        Returns:
            Full URL to retrieve the image.
        """
        params = f"filename={filename}&type={folder_type}"
        if subfolder:
            params += f"&subfolder={subfolder}"
        return f"{self._base_url}/view?{params}"

    async def get_image(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> bytes:
        """Download an image from ComfyUI.

        Args:
            filename: Image filename.
            subfolder: Subfolder within the type folder.
            folder_type: Folder type (output, input, temp).

        Returns:
            Raw image bytes.

        Raises:
            ComfyUIAPIError: If download fails.
        """
        try:
            params = {"filename": filename, "type": folder_type}
            if subfolder:
                params["subfolder"] = subfolder

            response = await self.client.get("/view", params=params)

            if response.status_code != 200:
                raise ComfyUIAPIError(
                    f"Failed to get image: {response.text}",
                    status_code=response.status_code,
                )

            return response.content

        except httpx.RequestError as e:
            logger.error("comfyui.get_image_failed", error=str(e))
            raise ComfyUIConnectionError(f"Failed to get image: {e}") from e
