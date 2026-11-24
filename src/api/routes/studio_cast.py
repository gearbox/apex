"""Studio cast — members, pinned reference images and looks (``/v1/studio/cast``, S1-P3).

Endpoints:
  GET    /v1/studio/cast                              — keyset page, newest first; ``?holding=<asset_ref>``
  POST   /v1/studio/cast                              — create with its first references (201)
  GET    /v1/studio/cast/{id}                         — member with references and looks
  PATCH  /v1/studio/cast/{id}                         — tri-state update
  DELETE /v1/studio/cast/{id}                         — delete (204; 409 cast_member_in_use for a performer)
  POST   /v1/studio/cast/{id}/references              — add a reference, pinning its asset (201)
  PATCH  /v1/studio/cast/{id}/references/{ref_id}     — relabel / swap the asset
  DELETE /v1/studio/cast/{id}/references/{ref_id}     — remove (204); positions are re-compacted
  PUT    /v1/studio/cast/{id}/references/order        — reorder (exact permutation, else 422 invalid_order)
  POST   /v1/studio/cast/{id}/looks                   — create a look (201)
  PATCH  /v1/studio/cast/{id}/looks/{look_id}         — rename / replace the reference set
  DELETE /v1/studio/cast/{id}/looks/{look_id}         — remove (204)

Mutations are legal-enforced like every other mutating route (none opts out). Errors are
:class:`~src.studio.errors.StudioError` subclasses rendered by one app-level handler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated
from uuid import UUID

from litestar import Controller, delete, get, patch, post, put
from litestar.di import Provide
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter
from litestar.status_codes import HTTP_201_CREATED, HTTP_204_NO_CONTENT

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.pagination import CursorPage
from src.api.schemas.studio import (
    CastLookIn,
    CastLookOut,
    CastLookPatch,
    CastMemberCreate,
    CastMemberOut,
    CastMemberPatch,
    CastMemberSummary,
    CastReferenceIn,
    CastReferenceOrder,
    CastReferenceOut,
    CastReferencePatch,
)
from src.api.security import auth_guard
from src.api.security.studio import studio_feature_guard
from src.core.library_ref import parse_asset_ref
from src.studio.errors import InvalidAssetRef
from src.studio.services.cast import CastService

if TYPE_CHECKING:
    from collections.abc import Sequence

_ERRORS = {
    400: ResponseSpec(
        data_container=ErrorEnvelope, description="invalid_cursor or invalid_asset_ref."
    ),
    404: ResponseSpec(
        data_container=ErrorEnvelope,
        description=(
            "cast_member_not_found, cast_reference_not_found, cast_look_not_found or "
            "asset_not_found (or studio disabled)."
        ),
    ),
    409: ResponseSpec(
        data_container=ErrorEnvelope,
        description=(
            "cast_name_taken, cast_label_taken, cast_look_name_taken, cast_limit_reached "
            "or cast_member_in_use."
        ),
    ),
    422: ResponseSpec(
        data_container=ErrorEnvelope,
        description="document_invalid, invalid_order or asset_not_eligible.",
    ),
}


class StudioCastController(Controller):
    """Cast CRUD: members, references (pinned library images) and looks."""

    path = "/v1/studio/cast"
    tags: Sequence[str] | None = ("Studio",)
    # The feature guard runs first: a disabled studio answers 404 even when unauthenticated.
    guards = [studio_feature_guard, auth_guard]  # noqa: RUF012
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    # -- members -------------------------------------------------------------------------------

    @get("/", responses=_ERRORS)
    async def list_cast(
        self,
        current_user_id: UUID,
        product_id: str,
        studio_cast_service: CastService,
        limit: Annotated[int, Parameter(ge=1, le=50)] = 30,
        cursor: str | None = None,
        holding: Annotated[
            str | None,
            Parameter(
                description=(
                    "An asset ref (`upload:<uuid>` / `output:<uuid>`): only the members that "
                    "reference that asset (to name the holders of a `409 asset_held`)."
                )
            ),
        ] = None,
    ) -> CursorPage[CastMemberSummary]:
        """Paginated list of the caller's cast members, newest first."""
        asset = None
        if holding is not None:
            try:
                asset = parse_asset_ref(holding)
            except ValueError as exc:
                raise InvalidAssetRef from exc
        return await studio_cast_service.list_page(
            current_user_id, product_id, limit=limit, cursor=cursor, holding=asset
        )

    @post("/", status_code=HTTP_201_CREATED, responses=_ERRORS)
    async def create_member(
        self,
        current_user_id: UUID,
        product_id: str,
        data: CastMemberCreate,
        studio_cast_service: CastService,
    ) -> CastMemberOut:
        """Create a cast member with its first references; each reference pins its asset."""
        return await studio_cast_service.create(current_user_id, product_id, data)

    @get("/{member_id:uuid}", responses=_ERRORS)
    async def get_member(
        self,
        current_user_id: UUID,
        product_id: str,
        member_id: UUID,
        studio_cast_service: CastService,
    ) -> CastMemberOut:
        """Fetch one member with its references (by position) and looks (by name)."""
        return await studio_cast_service.get(member_id, current_user_id, product_id)

    @patch("/{member_id:uuid}", responses=_ERRORS)
    async def patch_member(
        self,
        current_user_id: UUID,
        product_id: str,
        member_id: UUID,
        data: CastMemberPatch,
        studio_cast_service: CastService,
    ) -> CastMemberOut:
        """Tri-state update: omitted fields are kept, ``null`` clears description / appearance."""
        return await studio_cast_service.patch(member_id, current_user_id, product_id, data)

    @delete("/{member_id:uuid}", status_code=HTTP_204_NO_CONTENT, responses=_ERRORS)
    async def delete_member(
        self,
        current_user_id: UUID,
        product_id: str,
        member_id: UUID,
        studio_cast_service: CastService,
    ) -> None:
        """Delete a member and release every hold (409 ``cast_member_in_use`` for a performer)."""
        await studio_cast_service.delete(member_id, current_user_id, product_id)

    # -- references ----------------------------------------------------------------------------

    @post("/{member_id:uuid}/references", status_code=HTTP_201_CREATED, responses=_ERRORS)
    async def add_reference(
        self,
        current_user_id: UUID,
        product_id: str,
        member_id: UUID,
        data: CastReferenceIn,
        studio_cast_service: CastService,
    ) -> CastReferenceOut:
        """Append a reference (position = count) and pin its asset."""
        return await studio_cast_service.add_reference(member_id, current_user_id, product_id, data)

    @put("/{member_id:uuid}/references/order", responses=_ERRORS)
    async def reorder_references(
        self,
        current_user_id: UUID,
        product_id: str,
        member_id: UUID,
        data: CastReferenceOrder,
        studio_cast_service: CastService,
    ) -> CastMemberOut:
        """Reorder: ``reference_ids`` must be exactly the member's references, each once."""
        return await studio_cast_service.reorder_references(
            member_id, current_user_id, product_id, data
        )

    @patch("/{member_id:uuid}/references/{reference_id:uuid}", responses=_ERRORS)
    async def change_reference(
        self,
        current_user_id: UUID,
        product_id: str,
        member_id: UUID,
        reference_id: UUID,
        data: CastReferencePatch,
        studio_cast_service: CastService,
    ) -> CastReferenceOut:
        """Relabel a reference and / or swap its asset (the pin moves to the new asset)."""
        return await studio_cast_service.change_reference(
            member_id, reference_id, current_user_id, product_id, data
        )

    @delete(
        "/{member_id:uuid}/references/{reference_id:uuid}",
        status_code=HTTP_204_NO_CONTENT,
        responses=_ERRORS,
    )
    async def delete_reference(
        self,
        current_user_id: UUID,
        product_id: str,
        member_id: UUID,
        reference_id: UUID,
        studio_cast_service: CastService,
    ) -> None:
        """Remove a reference: it leaves every look, its asset is released, positions re-compact."""
        await studio_cast_service.delete_reference(
            member_id, reference_id, current_user_id, product_id
        )

    # -- looks ---------------------------------------------------------------------------------

    @post("/{member_id:uuid}/looks", status_code=HTTP_201_CREATED, responses=_ERRORS)
    async def create_look(
        self,
        current_user_id: UUID,
        product_id: str,
        member_id: UUID,
        data: CastLookIn,
        studio_cast_service: CastService,
    ) -> CastLookOut:
        """Create a look: a named set of at least one of the member's references."""
        return await studio_cast_service.create_look(member_id, current_user_id, product_id, data)

    @patch("/{member_id:uuid}/looks/{look_id:uuid}", responses=_ERRORS)
    async def change_look(
        self,
        current_user_id: UUID,
        product_id: str,
        member_id: UUID,
        look_id: UUID,
        data: CastLookPatch,
        studio_cast_service: CastService,
    ) -> CastLookOut:
        """Rename a look and / or replace its whole reference set (at least one)."""
        return await studio_cast_service.change_look(
            member_id, look_id, current_user_id, product_id, data
        )

    @delete(
        "/{member_id:uuid}/looks/{look_id:uuid}",
        status_code=HTTP_204_NO_CONTENT,
        responses=_ERRORS,
    )
    async def delete_look(
        self,
        current_user_id: UUID,
        product_id: str,
        member_id: UUID,
        look_id: UUID,
        studio_cast_service: CastService,
    ) -> None:
        """Remove a look (its references stay)."""
        await studio_cast_service.delete_look(member_id, look_id, current_user_id, product_id)


__all__ = ["StudioCastController"]
