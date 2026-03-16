"""Generation API routes."""

from collections.abc import Sequence
from typing import Annotated

import structlog
from litestar import Controller, Response, get, post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.status_codes import HTTP_201_CREATED

from src.api.schemas.generation import (
    HealthResponse,
    ImageUploadResponse,
)
from src.api.security import auth_guard
from src.api.services.comfyui_client import ComfyUIClient, ComfyUIClientError
from src.core.uid import new_id

logger = structlog.get_logger(__name__)


class HealthController(Controller):
    """Health check endpoints."""

    path = "/health"
    tags: Sequence[str] | None = ["Health"]

    @get("/")
    async def health_check(
        self,
        comfyui_client: ComfyUIClient,
    ) -> HealthResponse:
        """Check API and ComfyUI connectivity.

        Returns health status of the service and its dependencies.
        """
        comfyui_connected = await comfyui_client.health_check()

        return HealthResponse(
            status="healthy" if comfyui_connected else "unhealthy",
            comfyui_connected=comfyui_connected,
        )


class ImageController(Controller):
    """Image upload and retrieval endpoints."""

    path = "/v1/images"
    tags: Sequence[str] | None = ["Images"]
    guards = [auth_guard]

    @post("/upload")
    async def upload_image(
        self,
        comfyui_client: ComfyUIClient,
        data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
    ) -> Response[ImageUploadResponse]:
        """Upload an image to ComfyUI for use in generation.

        Returns the filename that can be referenced in generation requests.
        """
        try:
            image_data = await data.read()

            # Generate unique filename preserving extension
            ext = data.filename.rsplit(".", 1)[-1] if data.filename else "png"
            logger.debug("image.uploading", ext=ext)
            unique_filename = f"upload_{new_id().hex[:12]}.{ext}"

            result = await comfyui_client.upload_image(image_data, unique_filename)

            return Response(
                content=ImageUploadResponse(
                    filename=result.get("name", unique_filename),
                    subfolder=result.get("subfolder", ""),
                    type=result.get("type", "input"),
                ),
                status_code=HTTP_201_CREATED,
            )

        except ComfyUIClientError as e:
            logger.error("image.upload_failed", error=str(e))
            return Response(
                content=ImageUploadResponse(
                    filename="",
                    subfolder="",
                    type="input",
                ),
                status_code=500,
            )
