"""ComfyUI HTTP client service for API communication."""

import logging
from typing import Any

import httpx

from src.core.config import Settings

logger = logging.getLogger(__name__)


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
            logger.info(f"ComfyUI client connected to {self._base_url}")

    async def close(self) -> None:
        """Close HTTP client connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("ComfyUI client disconnected")

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
            logger.warning(f"ComfyUI health check failed: {e}")
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
                logger.error(f"ComfyUI queue_prompt failed: {response.status_code} - {error_text}")
                raise ComfyUIAPIError(
                    f"Failed to queue prompt: {error_text}",
                    status_code=response.status_code,
                )

            result = response.json()
            logger.debug(f"Prompt queued successfully: {result.get('prompt_id')}")
            return result

        except httpx.RequestError as e:
            logger.error(f"ComfyUI connection error: {e}")
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

            return response.json() if response.status_code == 200 else {}
        except httpx.RequestError as e:
            logger.warning(f"Failed to get history for {prompt_id}: {e}")
            return {}

    async def get_queue(self) -> dict[str, Any]:
        """Get current ComfyUI queue status.

        Returns:
            Queue info with running and pending items.
        """
        try:
            response = await self.client.get("/queue")
            if response.status_code == 200:
                return response.json()
            return {"queue_running": [], "queue_pending": []}
        except httpx.RequestError as e:
            logger.warning(f"Failed to get queue: {e}")
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

            result = response.json()
            logger.debug(f"Image uploaded: {result.get('name')}")
            return result

        except httpx.RequestError as e:
            logger.error(f"Failed to upload image: {e}")
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
            logger.error(f"Failed to get image: {e}")
            raise ComfyUIConnectionError(f"Failed to get image: {e}") from e
