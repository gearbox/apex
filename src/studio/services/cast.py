"""Cast: members, their pinned reference images and their looks (``/v1/studio/cast``, S1-P3).

A cast member is a person, character or performer the user keeps references for. Each reference
is a user-labelled library image; **it pins its asset** through the library's hold mechanism
(``AssetHoldService``): the asset is exempt from retention and cannot be deleted for as long as
the reference exists, and its retention clock restarts when the last hold goes. The reference
stores no ``asset_ref`` of its own - the asset lives on the hold row (one source of truth) and is
read through a join.

Transactions and locks
----------------------
The service never commits or flushes; it runs inside the request transaction and calls the
library's ``AssetHoldService`` in the same one. **Lock order: artist → cast member → reference /
look → (assets, inside the hold service).** Nothing takes them the other way round.

* Every member-level mutation (references, looks, order, PATCH, delete) first locks the member
  ``FOR UPDATE``. Counts of references and looks are taken under that lock, so those caps are
  hard (two parallel adds at 11 references: one wins, one gets ``409 cast_limit_reached``). The
  member cap works like ``MAX_ARTISTS_PER_OWNER``: a soft cap, counted without a lock.
* A multi-step change is one savepoint (hold → insert, hold → repoint → release): a failure
  leaves neither a hold without its reference nor a reference without its hold. Invariant after
  every committed transaction: every reference has exactly one hold, and every
  ``studio.cast_reference`` hold has exactly one reference.

Validation follows RD6: text is NFC-normalized, trimmed and (for names and labels) whitespace-
collapsed, then every length is re-checked **after** normalization by ``collect_text_issues``
against the limits declared on the wire Structs - the only source of the limits, so there are no
copies here. Logs carry ids and counts only: never names, labels, descriptions or appearance.
"""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING, Final

import msgspec
import structlog
from sqlalchemy.exc import IntegrityError

from src.api.schemas.pagination import CursorPage, decode_cursor, encode_cursor
from src.api.schemas.studio import (
    CastLookOut,
    CastMemberOut,
    CastMemberSummary,
    CastReferenceOut,
)
from src.api.services.library_holds import (
    AssetHolder,
    AssetNotEligibleError,
    AssetNotFoundError,
)
from src.api.services.library_media import build_media_object
from src.core.enums import MediaKind
from src.core.library_ref import AssetRef, LibraryAssetSource, format_asset_ref, parse_asset_ref
from src.core.uid import new_id
from src.db.integrity import violated_constraint
from src.db.models.studio import CAST_PERFORMER_FK
from src.db.models.studio_cast import (
    CAST_REFERENCE_HOLDER_KIND,
    LOOK_NAME_UNIQUE,
    MEMBER_NAME_UNIQUE,
    REFERENCE_LABEL_UNIQUE,
)
from src.db.repositories.output import OutputRepository
from src.db.repositories.studio_cast import (
    CastReferenceRow,
    StudioCastLookRepository,
    StudioCastMemberRepository,
    StudioCastReferenceRepository,
)
from src.db.repositories.user_image import UserImageRepository
from src.studio.constraints import (
    MAX_CAST_LOOKS,
    MAX_CAST_MEMBERS_PER_OWNER,
    MAX_CAST_REFERENCES,
    collect_text_issues,
)
from src.studio.errors import (
    AssetNotEligible,
    AssetNotFound,
    CastLabelTaken,
    CastLimitReached,
    CastLookNameTaken,
    CastLookNotFound,
    CastMemberInUse,
    CastMemberNotFound,
    CastNameTaken,
    CastReferenceNotFound,
    DocumentIssue,
    InvalidCursor,
    InvalidReferenceOrder,
    StudioDocumentInvalid,
    StudioError,
)
from src.studio.validation import normalize_block

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.media import MediaObject
    from src.api.schemas.studio import (
        CastLookIn,
        CastLookPatch,
        CastMemberCreate,
        CastMemberPatch,
        CastReferenceIn,
        CastReferenceOrder,
        CastReferencePatch,
    )
    from src.api.services.library_holds import AssetHoldService
    from src.db.models.studio_cast import StudioCastLook, StudioCastMember

__all__ = ["CastService"]

logger = structlog.get_logger(__name__)

_CAST_REFERENCE_ACCEPTS: Final = frozenset({MediaKind.IMAGE})
"""A cast reference pins images only (S4 feeds them to image models)."""

_EMPTY: Final = "empty_after_normalization"
_INVALID_ASSET: Final = "invalid_asset_ref"

_NAME_CONFLICTS: Final[dict[str, type[StudioError]]] = {
    MEMBER_NAME_UNIQUE: CastNameTaken,
    REFERENCE_LABEL_UNIQUE: CastLabelTaken,
    LOOK_NAME_UNIQUE: CastLookNameTaken,
}
_KNOWN_CONSTRAINTS: Final = frozenset(_NAME_CONFLICTS)

type _MediaKey = tuple[str, UUID]


def _tidy(raw: str) -> str:
    """NFC, trimmed, inner whitespace collapsed: for names, labels and look names."""
    return " ".join(unicodedata.normalize("NFC", raw).split())


def _name_conflict(exc: IntegrityError) -> StudioError | None:
    """The typed 409 for a unique-index violation of a name or label; ``None`` if it is another."""
    constraint = violated_constraint(exc, _KNOWN_CONSTRAINTS)
    return None if constraint is None else _NAME_CONFLICTS[constraint]()


class CastService:
    """Business logic for the cast. Methods take the owner scope explicitly."""

    def __init__(self, session: AsyncSession, holds: AssetHoldService) -> None:
        self._session = session
        self._holds = holds
        self._members = StudioCastMemberRepository(session)
        self._refs = StudioCastReferenceRepository(session)
        self._looks = StudioCastLookRepository(session)

    # ----------------------------------------------------------------------------------------
    # Reads
    # ----------------------------------------------------------------------------------------

    async def list_page(
        self,
        user_id: UUID,
        product_id: str,
        *,
        limit: int,
        cursor: str | None,
        holding: AssetRef | None = None,
    ) -> CursorPage[CastMemberSummary]:
        """The caller's members, newest first; ``holding`` keeps those that reference that asset."""
        try:
            cursor_ts, cursor_id = decode_cursor(cursor) if cursor is not None else (None, None)
        except ValueError as exc:
            raise InvalidCursor from exc
        rows = await self._members.list_page(
            user_id=user_id,
            product_id=product_id,
            limit=limit,
            cursor_ts=cursor_ts,
            cursor_id=cursor_id,
            holding=holding,
        )
        has_more = len(rows) > limit
        page = list(rows[:limit])
        ids = [member.id for member in page]
        reference_counts = await self._refs.counts_for_members(ids)
        look_counts = await self._looks.counts_for_members(ids)
        covers = await self._refs.list_covers(ids, user_id=user_id, product_id=product_id)
        media = await self._media(list(covers.values()), user_id)
        items = [
            CastMemberSummary(
                id=member.id,
                name=member.name,
                appearance=member.appearance,
                reference_count=reference_counts.get(member.id, 0),
                look_count=look_counts.get(member.id, 0),
                cover=(
                    media[(covers[member.id].source.value, covers[member.id].asset_id)]
                    if member.id in covers
                    else None
                ),
                created_at=member.created_at,
                updated_at=member.updated_at,
            )
            for member in page
        ]
        next_cursor = encode_cursor(page[-1].created_at, page[-1].id) if has_more and page else None
        return CursorPage(items=items, limit=limit, has_more=has_more, next_cursor=next_cursor)

    async def get(self, member_id: UUID, user_id: UUID, product_id: str) -> CastMemberOut:
        member = await self._members.get(member_id, user_id=user_id, product_id=product_id)
        if member is None:
            raise CastMemberNotFound
        return await self._detail(member, user_id, product_id)

    # ----------------------------------------------------------------------------------------
    # Members
    # ----------------------------------------------------------------------------------------

    async def create(self, user_id: UUID, product_id: str, data: CastMemberCreate) -> CastMemberOut:
        """Create a member with its first references, atomically.

        Raises:
            StudioDocumentInvalid: With every issue found (bad text, bad asset refs).
            CastLimitReached: The caller already has the maximum number of members (a soft cap).
            CastNameTaken: The name collides case-insensitively for this owner.
            CastLabelTaken: Two initial references share a label (case-insensitively).
            AssetNotFound: A referenced asset is missing / not the caller's.
            AssetNotEligible: A referenced asset is a derivative, not an image, or its job is
                not available.
        """
        issues: list[DocumentIssue] = []
        name = self._text(data.name, "name", issues)
        description = normalize_block(data.description)
        appearance = normalize_block(data.appearance)
        references: list[tuple[str, AssetRef | None]] = []
        for index, reference in enumerate(data.references):
            label = self._text(reference.label, f"references[{index}].label", issues)
            asset = self._asset(reference.asset, f"references[{index}].asset", issues)
            references.append((label, asset))
        # RD6: NFC can expand a value past the limit the wire already checked.
        collect_text_issues(
            msgspec.structs.replace(
                data,
                name=name,
                description=description,
                appearance=appearance,
                references=[
                    msgspec.structs.replace(reference, label=label)
                    for reference, (label, _) in zip(data.references, references, strict=True)
                ],
            ),
            issues,
        )
        if issues:
            raise StudioDocumentInvalid(tuple(issues))

        # A soft cap, like the artists': two simultaneous creates at 49 may both pass. It bounds
        # abuse, it is not an integrity rule, so no lock is taken.
        if (
            await self._members.count_for_owner(user_id=user_id, product_id=product_id)
            >= MAX_CAST_MEMBERS_PER_OWNER
        ):
            raise CastLimitReached("members", MAX_CAST_MEMBERS_PER_OWNER)

        try:
            async with self._session.begin_nested():
                member = await self._members.insert(
                    user_id=user_id,
                    product_id=product_id,
                    name=name,
                    description=description,
                    appearance=appearance,
                )
                for position, (label, asset) in enumerate(references):
                    # ``asset`` is None only when an issue was recorded, which raised above.
                    await self._add_reference(
                        member.id, user_id, product_id, label=label, asset=asset, position=position
                    )
        except IntegrityError as exc:
            conflict = _name_conflict(exc)
            if conflict is None:
                raise
            raise conflict from exc
        logger.info(
            "studio.cast_member_created",
            cast_member_id=str(member.id),
            user_id=str(user_id),
            reference_count=len(references),
        )
        return await self._detail(member, user_id, product_id)

    async def patch(
        self, member_id: UUID, user_id: UUID, product_id: str, data: CastMemberPatch
    ) -> CastMemberOut:
        """Tri-state update of ``name`` / ``description`` / ``appearance`` under the member lock."""
        issues: list[DocumentIssue] = []
        changes: dict[str, object] = {}
        if data.name is not msgspec.UNSET:
            changes["name"] = self._text(data.name, "name", issues)
        if data.description is not msgspec.UNSET:
            changes["description"] = normalize_block(data.description)
        if data.appearance is not msgspec.UNSET:
            changes["appearance"] = normalize_block(data.appearance)
        # RD6: only the fields the client sent are re-checked; stored values were checked on write.
        collect_text_issues(msgspec.structs.replace(data, **changes), issues)
        if issues:
            raise StudioDocumentInvalid(tuple(issues))

        await self._lock_member(member_id, user_id, product_id)
        try:
            async with self._session.begin_nested():
                updated = await self._members.update(
                    member_id, user_id=user_id, product_id=product_id, changes=changes
                )
        except IntegrityError as exc:
            conflict = _name_conflict(exc)
            if conflict is None:
                raise
            raise conflict from exc
        if updated is None:  # pragma: no cover - the row lock above rules out a concurrent delete
            raise CastMemberNotFound
        logger.info(
            "studio.cast_member_updated",
            cast_member_id=str(member_id),
            user_id=str(user_id),
            fields=[f for f in data.__struct_fields__ if getattr(data, f) is not msgspec.UNSET],
        )
        return await self._detail(updated, user_id, product_id)

    async def delete(self, member_id: UUID, user_id: UUID, product_id: str) -> None:
        """Delete a member: its references, looks and links cascade, then every hold is released.

        Refused (``409 cast_member_in_use``) while an artist names the member as its performer:
        silently unlinking would change the artist's look without telling the user. An artist
        PATCH that races the delete is caught by the foreign key and ends in the same 409 (or, if
        the delete won, in the PATCH's 404) - never a 500.

        Raises:
            CastMemberNotFound: No such member for the caller.
            CastMemberInUse: An artist uses the member as its performer.
        """
        await self._lock_member(member_id, user_id, product_id)
        in_use = await self._members.count_performed_artists(
            member_id, user_id=user_id, product_id=product_id
        )
        if in_use:
            raise CastMemberInUse(in_use)
        hold_ids = await self._refs.hold_ids_of_member(member_id)
        try:
            async with self._session.begin_nested():
                deleted = await self._members.delete(
                    member_id, user_id=user_id, product_id=product_id
                )
        except IntegrityError as exc:
            # An artist PATCH linked the member after the count: the FK is the backstop. If the
            # recount still finds none, the failure is unexplained -- surface it rather than
            # blame an artist that is not there with a misleading 409.
            if violated_constraint(exc, frozenset({CAST_PERFORMER_FK})) is None:
                raise
            in_use = await self._members.count_performed_artists(
                member_id, user_id=user_id, product_id=product_id
            )
            if not in_use:
                raise
            raise CastMemberInUse(in_use) from exc
        if not deleted:  # pragma: no cover - the row lock above rules out a concurrent delete
            raise CastMemberNotFound
        # The cascade removed the references, which is what freed their holds to be deleted.
        await self._holds.release(hold_ids)
        logger.info(
            "studio.cast_member_deleted",
            cast_member_id=str(member_id),
            user_id=str(user_id),
            reference_count=len(hold_ids),
        )

    # ----------------------------------------------------------------------------------------
    # References
    # ----------------------------------------------------------------------------------------

    async def add_reference(
        self,
        member_id: UUID,
        user_id: UUID,
        product_id: str,
        data: CastReferenceIn,
    ) -> CastReferenceOut:
        """Append a reference (``position = count``) and pin its asset."""
        issues: list[DocumentIssue] = []
        label = self._text(data.label, "label", issues)
        asset = self._asset(data.asset, "asset", issues)
        collect_text_issues(msgspec.structs.replace(data, label=label), issues)
        if issues:
            raise StudioDocumentInvalid(tuple(issues))

        await self._lock_member(member_id, user_id, product_id)
        count = await self._refs.count_for_member(member_id)
        if count >= MAX_CAST_REFERENCES:
            raise CastLimitReached("references", MAX_CAST_REFERENCES)
        try:
            async with self._session.begin_nested():
                reference_id = await self._add_reference(
                    member_id, user_id, product_id, label=label, asset=asset, position=count
                )
        except IntegrityError as exc:
            conflict = _name_conflict(exc)
            if conflict is None:
                raise
            raise conflict from exc
        await self._members.touch(member_id)
        logger.info(
            "studio.cast_reference_added",
            cast_member_id=str(member_id),
            reference_id=str(reference_id),
            user_id=str(user_id),
        )
        return await self._reference_by_id(reference_id, member_id, user_id, product_id)

    async def change_reference(
        self,
        member_id: UUID,
        reference_id: UUID,
        user_id: UUID,
        product_id: str,
        data: CastReferencePatch,
    ) -> CastReferenceOut:
        """Relabel a reference and / or point it at another asset.

        Changing the asset moves the pin in one savepoint: hold the new asset, repoint
        ``hold_id``, release the old hold (which restarts the old asset's retention clock if that
        was its last hold). The same asset again changes nothing (200).
        """
        issues: list[DocumentIssue] = []
        changes: dict[str, object] = {}
        asset: AssetRef | None = None
        if data.label is not msgspec.UNSET:
            changes["label"] = self._text(data.label, "label", issues)
        if data.asset is not msgspec.UNSET:
            asset = self._asset(data.asset, "asset", issues)
        collect_text_issues(msgspec.structs.replace(data, **changes), issues)
        if issues:
            raise StudioDocumentInvalid(tuple(issues))

        await self._lock_member(member_id, user_id, product_id)
        reference = await self._refs.get(
            reference_id, member_id, user_id=user_id, product_id=product_id
        )
        if reference is None:
            raise CastReferenceNotFound
        if "label" in changes and changes["label"] == reference.label:
            del changes["label"]
        old_hold_id: UUID | None = None
        if asset is not None and asset != AssetRef(
            source=reference.source, asset_id=reference.asset_id
        ):
            old_hold_id = reference.hold_id
        if not changes and old_hold_id is None:
            return await self._reference_by_id(reference_id, member_id, user_id, product_id)

        try:
            async with self._session.begin_nested():
                if asset is not None and old_hold_id is not None:
                    changes["hold_id"] = await self._hold(
                        asset, reference_id, user_id=user_id, product_id=product_id
                    )
                await self._refs.update(
                    reference_id,
                    member_id,
                    user_id=user_id,
                    product_id=product_id,
                    changes=changes,
                )
                if old_hold_id is not None:
                    await self._holds.release([old_hold_id])
        except IntegrityError as exc:
            conflict = _name_conflict(exc)
            if conflict is None:
                raise
            raise conflict from exc
        await self._members.touch(member_id)
        logger.info(
            "studio.cast_reference_changed",
            cast_member_id=str(member_id),
            reference_id=str(reference_id),
            user_id=str(user_id),
            asset_changed=old_hold_id is not None,
        )
        return await self._reference_by_id(reference_id, member_id, user_id, product_id)

    async def delete_reference(
        self, member_id: UUID, reference_id: UUID, user_id: UUID, product_id: str
    ) -> None:
        """Delete a reference (it leaves every look), release its hold, re-compact positions."""
        await self._lock_member(member_id, user_id, product_id)
        hold_id = await self._refs.delete(
            reference_id, member_id, user_id=user_id, product_id=product_id
        )
        if hold_id is None:
            raise CastReferenceNotFound
        await self._holds.release([hold_id])
        await self._refs.compact_positions(member_id)
        await self._members.touch(member_id)
        logger.info(
            "studio.cast_reference_removed",
            cast_member_id=str(member_id),
            reference_id=str(reference_id),
            user_id=str(user_id),
        )

    async def reorder_references(
        self,
        member_id: UUID,
        user_id: UUID,
        product_id: str,
        data: CastReferenceOrder,
    ) -> CastMemberOut:
        """Set the references' order; the ids must be exactly the member's, each once."""
        member = await self._lock_member(member_id, user_id, product_id)
        current = await self._refs.ids_of_member(member_id)
        requested = list(data.reference_ids)
        if len(requested) != len(set(requested)) or set(requested) != set(current):
            raise InvalidReferenceOrder
        await self._refs.set_order(member_id, requested)
        await self._members.touch(member_id)
        return await self._detail_fresh(member, user_id, product_id)

    # ----------------------------------------------------------------------------------------
    # Looks
    # ----------------------------------------------------------------------------------------

    async def create_look(
        self, member_id: UUID, user_id: UUID, product_id: str, data: CastLookIn
    ) -> CastLookOut:
        """Create a look: a named set of at least one of the member's references."""
        issues: list[DocumentIssue] = []
        name = self._text(data.name, "name", issues)
        collect_text_issues(msgspec.structs.replace(data, name=name), issues)
        self._shape_issues(data.reference_ids, issues)

        await self._lock_member(member_id, user_id, product_id)
        await self._membership_issues(member_id, data.reference_ids, issues)
        if issues:
            raise StudioDocumentInvalid(tuple(issues))
        if await self._looks.count_for_member(member_id) >= MAX_CAST_LOOKS:
            raise CastLimitReached("looks", MAX_CAST_LOOKS)
        try:
            async with self._session.begin_nested():
                look = await self._looks.insert(
                    member_id=member_id, user_id=user_id, product_id=product_id, name=name
                )
                await self._looks.replace_links(look.id, member_id, data.reference_ids)
        except IntegrityError as exc:
            conflict = _name_conflict(exc)
            if conflict is None:
                raise
            raise conflict from exc
        await self._members.touch(member_id)
        logger.info(
            "studio.cast_look_saved",
            cast_member_id=str(member_id),
            look_id=str(look.id),
            user_id=str(user_id),
            reference_count=len(data.reference_ids),
        )
        return await self._look_out(look)

    async def change_look(
        self,
        member_id: UUID,
        look_id: UUID,
        user_id: UUID,
        product_id: str,
        data: CastLookPatch,
    ) -> CastLookOut:
        """Rename a look and / or replace its whole reference set (at least one)."""
        issues: list[DocumentIssue] = []
        changes: dict[str, object] = {}
        if data.name is not msgspec.UNSET:
            changes["name"] = self._text(data.name, "name", issues)
        collect_text_issues(msgspec.structs.replace(data, **changes), issues)
        reference_ids = None if data.reference_ids is msgspec.UNSET else list(data.reference_ids)
        if reference_ids is not None:
            self._shape_issues(reference_ids, issues)

        await self._lock_member(member_id, user_id, product_id)
        look = await self._looks.get(look_id, member_id, user_id=user_id, product_id=product_id)
        if look is None:
            raise CastLookNotFound
        if reference_ids is not None:
            await self._membership_issues(member_id, reference_ids, issues)
        if issues:
            raise StudioDocumentInvalid(tuple(issues))
        try:
            async with self._session.begin_nested():
                if "name" in changes:
                    await self._looks.rename(
                        look_id,
                        member_id,
                        user_id=user_id,
                        product_id=product_id,
                        name=str(changes["name"]),
                    )
                if reference_ids is not None:
                    await self._looks.replace_links(look_id, member_id, reference_ids)
                await self._looks.touch(look_id)
        except IntegrityError as exc:
            conflict = _name_conflict(exc)
            if conflict is None:
                raise
            raise conflict from exc
        await self._members.touch(member_id)
        logger.info(
            "studio.cast_look_saved",
            cast_member_id=str(member_id),
            look_id=str(look_id),
            user_id=str(user_id),
            reference_count=None if reference_ids is None else len(reference_ids),
        )
        fresh = await self._looks.get(look_id, member_id, user_id=user_id, product_id=product_id)
        if fresh is None:  # pragma: no cover - the member lock rules out a concurrent delete
            raise CastLookNotFound
        return await self._look_out(fresh)

    async def delete_look(
        self, member_id: UUID, look_id: UUID, user_id: UUID, product_id: str
    ) -> None:
        await self._lock_member(member_id, user_id, product_id)
        if not await self._looks.delete(look_id, member_id, user_id=user_id, product_id=product_id):
            raise CastLookNotFound
        await self._members.touch(member_id)
        logger.info(
            "studio.cast_look_deleted",
            cast_member_id=str(member_id),
            look_id=str(look_id),
            user_id=str(user_id),
        )

    # ----------------------------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------------------------

    async def _lock_member(
        self, member_id: UUID, user_id: UUID, product_id: str
    ) -> StudioCastMember:
        member = await self._members.lock(member_id, user_id=user_id, product_id=product_id)
        if member is None:
            raise CastMemberNotFound
        return member

    @staticmethod
    def _text(raw: str, path: str, issues: list[DocumentIssue]) -> str:
        """A name / label, normalized; ``""`` after recording that nothing is left of it."""
        text = _tidy(raw)
        if not text:
            issues.append(DocumentIssue(path, _EMPTY))
        return text

    @staticmethod
    def _asset(raw: str, path: str, issues: list[DocumentIssue]) -> AssetRef | None:
        try:
            return parse_asset_ref(raw)
        except ValueError:
            issues.append(DocumentIssue(path, _INVALID_ASSET))
            return None

    @staticmethod
    def _shape_issues(reference_ids: Sequence[UUID], issues: list[DocumentIssue]) -> None:
        """A look needs at least one reference, each named once (no member data needed)."""
        if not reference_ids:
            issues.append(DocumentIssue("reference_ids", "empty"))
        seen: set[UUID] = set()
        for index, reference_id in enumerate(reference_ids):
            if reference_id in seen:
                issues.append(DocumentIssue(f"reference_ids[{index}]", "duplicate_reference"))
            seen.add(reference_id)

    async def _membership_issues(
        self, member_id: UUID, reference_ids: Sequence[UUID], issues: list[DocumentIssue]
    ) -> None:
        """Every id must be a reference of **this** member (under the member lock)."""
        own = set(await self._refs.ids_of_member(member_id))
        issues.extend(
            DocumentIssue(f"reference_ids[{index}]", "reference_not_found")
            for index, reference_id in enumerate(reference_ids)
            if reference_id not in own
        )

    async def _hold(
        self, asset: AssetRef, reference_id: UUID, *, user_id: UUID, product_id: str
    ) -> UUID:
        """Pin ``asset`` for the reference; the library's errors become the studio's."""
        try:
            return await self._holds.hold(
                asset,
                holder=AssetHolder(kind=CAST_REFERENCE_HOLDER_KIND, id=reference_id),
                user_id=user_id,
                product_id=product_id,
                accept=_CAST_REFERENCE_ACCEPTS,
            )
        except AssetNotFoundError as exc:
            raise AssetNotFound from exc
        except AssetNotEligibleError as exc:
            raise AssetNotEligible(exc.reason.value) from exc

    async def _add_reference(
        self,
        member_id: UUID,
        user_id: UUID,
        product_id: str,
        *,
        label: str,
        asset: AssetRef | None,
        position: int,
    ) -> UUID:
        """Hold the asset, then insert the reference that owns the hold. Call in a savepoint.

        The reference's id is generated first so the hold can name it (``holder_id``).
        """
        if asset is None:  # pragma: no cover - callers raise on the recorded issue first
            raise StudioDocumentInvalid((DocumentIssue("asset", _INVALID_ASSET),))
        reference_id = new_id()
        hold_id = await self._hold(asset, reference_id, user_id=user_id, product_id=product_id)
        await self._refs.insert(
            reference_id=reference_id,
            member_id=member_id,
            user_id=user_id,
            product_id=product_id,
            label=label,
            position=position,
            hold_id=hold_id,
        )
        return reference_id

    async def _media(
        self, rows: Collection[CastReferenceRow], user_id: UUID
    ) -> dict[_MediaKey, MediaObject]:
        """``MediaObject`` per ``(source, asset_id)``, through the library's one builder."""
        upload_ids = sorted({r.asset_id for r in rows if r.source == LibraryAssetSource.UPLOAD})
        output_ids = sorted({r.asset_id for r in rows if r.source == LibraryAssetSource.OUTPUT})
        images = UserImageRepository(self._session)
        outputs = OutputRepository(self._session)
        uploads = await images.get_many(upload_ids, user_id=user_id)
        upload_variants = await images.batch_derivatives(upload_ids)
        generated = await outputs.get_many(output_ids, user_id=user_id)
        output_variants = await outputs.batch_derivatives(output_ids)
        media: dict[_MediaKey, MediaObject] = {}
        for row in rows:
            asset = (uploads if row.source == LibraryAssetSource.UPLOAD else generated).get(
                row.asset_id
            )
            if asset is None:
                # A hold keeps its asset alive (NO ACTION foreign key): this cannot happen.
                raise RuntimeError("A cast reference points at a library asset that is gone.")
            variants = (
                upload_variants if row.source == LibraryAssetSource.UPLOAD else output_variants
            ).get(row.asset_id, [])
            media[(row.source.value, row.asset_id)] = build_media_object(
                source=row.source,
                asset_id=asset.id,
                width=asset.width,
                height=asset.height,
                content_type=asset.content_type,
                size_bytes=asset.size_bytes,
                duration_ms=asset.duration_ms,
                derivatives=variants,
            )
        return media

    async def _reference_by_id(
        self, reference_id: UUID, member_id: UUID, user_id: UUID, product_id: str
    ) -> CastReferenceOut:
        row = await self._refs.get(reference_id, member_id, user_id=user_id, product_id=product_id)
        if row is None:  # pragma: no cover - the member lock rules out a concurrent delete
            raise CastReferenceNotFound
        media = await self._media([row], user_id)
        return self._build_reference(row, media)

    @staticmethod
    def _build_reference(
        row: CastReferenceRow, media: dict[_MediaKey, MediaObject]
    ) -> CastReferenceOut:
        return CastReferenceOut(
            id=row.id,
            label=row.label,
            position=row.position,
            asset=format_asset_ref(row.source, row.asset_id),
            media=media[(row.source.value, row.asset_id)],
        )

    async def _look_out(self, look: StudioCastLook) -> CastLookOut:
        links = await self._looks.links_for_looks([look.id])
        return CastLookOut(id=look.id, name=look.name, reference_ids=links.get(look.id, []))

    async def _detail_fresh(
        self, member: StudioCastMember, user_id: UUID, product_id: str
    ) -> CastMemberOut:
        """Detail of a member re-read after ``touch`` (``updated_at`` is changed by a Core UPDATE)."""
        fresh = await self._members.get(member.id, user_id=user_id, product_id=product_id)
        if fresh is None:  # pragma: no cover - the member lock rules out a concurrent delete
            raise CastMemberNotFound
        return await self._detail(fresh, user_id, product_id)

    async def _detail(
        self, member: StudioCastMember, user_id: UUID, product_id: str
    ) -> CastMemberOut:
        rows = await self._refs.list_for_members(
            [member.id], user_id=user_id, product_id=product_id
        )
        looks = await self._looks.list_for_members(
            [member.id], user_id=user_id, product_id=product_id
        )
        links = await self._looks.links_for_looks([look.id for look in looks])
        media = await self._media(rows, user_id)
        return CastMemberOut(
            id=member.id,
            name=member.name,
            description=member.description,
            appearance=member.appearance,
            references=[self._build_reference(row, media) for row in rows],
            looks=[
                CastLookOut(id=look.id, name=look.name, reference_ids=links.get(look.id, []))
                for look in looks
            ],
            created_at=member.created_at,
            updated_at=member.updated_at,
        )
