"""A ready-made studio graph for integration tests: user, library project, services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

from src.api.schemas.studio import (
    LyricsVersionCreate,
    PlanVersionCreate,
    ProjectProfilePut,
    SunoImportOut,
    SunoImportRequest,
    VersionCreateOut,
)
from src.core.enums import (
    StudioDocumentOrigin,
    StudioIncompatibleLyricsPolicy,
    StudioReleaseKind,
)
from src.db.models.studio import StudioDocumentVersion, StudioTrack
from src.studio.services.artist import ArtistService
from src.studio.services.export import ExportService
from src.studio.services.importer import SunoImportService
from src.studio.services.project import StudioProjectService
from src.studio.services.track import TrackService
from src.studio.services.version import DocumentVersionService

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.library import LibraryProject
    from src.db.models.user import User
    from src.studio.documents import LyricsDocument, PlanDocument


@dataclass
class StudioWorld:
    """One owner, one studio-enabled library project and the per-request services."""

    session: AsyncSession
    user: User
    library_project: LibraryProject
    product_id: str = "vex"
    artists: ArtistService = field(init=False)
    projects: StudioProjectService = field(init=False)
    tracks: TrackService = field(init=False)
    versions: DocumentVersionService = field(init=False)
    exports: ExportService = field(init=False)
    imports: SunoImportService = field(init=False)

    def __post_init__(self) -> None:
        self.artists = ArtistService(self.session)
        self.projects = StudioProjectService(self.session)
        self.tracks = TrackService(self.session)
        self.versions = DocumentVersionService(self.session)
        self.exports = ExportService(self.session)
        self.imports = SunoImportService(self.session, versions=self.versions)

    @property
    def user_id(self) -> UUID:
        return self.user.id

    @property
    def project_id(self) -> UUID:
        return self.library_project.id

    async def enable(
        self, *, artist_id: UUID | None = None, kind: StudioReleaseKind = StudioReleaseKind.ALBUM
    ) -> None:
        await self.projects.upsert_profile(
            self.project_id,
            self.user_id,
            self.product_id,
            ProjectProfilePut(release_kind=kind, artist_id=artist_id),
        )

    async def new_track(self, title: str = "Track") -> UUID:
        return (await self.tracks.create(self.project_id, self.user_id, self.product_id, title)).id

    async def plan(
        self,
        track_id: UUID,
        document: PlanDocument,
        *,
        parent: UUID | None = None,
        make_current: bool = True,
        incompatible: StudioIncompatibleLyricsPolicy = StudioIncompatibleLyricsPolicy.REJECT,
        origin: StudioDocumentOrigin = StudioDocumentOrigin.USER,
    ) -> tuple[VersionCreateOut, bool]:
        return await self.versions.create(
            track_id,
            PlanVersionCreate(
                content=document,
                parent_version_id=parent,
                make_current=make_current,
                incompatible_lyrics=incompatible,
            ),
            self.user_id,
            self.product_id,
            origin=origin,
        )

    async def lyrics(
        self,
        track_id: UUID,
        document: LyricsDocument,
        based_on: UUID,
        *,
        parent: UUID | None = None,
        make_current: bool = True,
        origin: StudioDocumentOrigin = StudioDocumentOrigin.USER,
    ) -> tuple[VersionCreateOut, bool]:
        return await self.versions.create(
            track_id,
            LyricsVersionCreate(
                content=document,
                based_on_plan_version_id=based_on,
                parent_version_id=parent,
                make_current=make_current,
            ),
            self.user_id,
            self.product_id,
            origin=origin,
        )

    async def import_suno(self, track_id: UUID, **fields: Any) -> SunoImportOut:
        """Import pasted Suno text (``style=``, ``exclude=``, ``lyrics=``, ``dry_run=``...)."""
        return await self.imports.import_suno(
            track_id, SunoImportRequest(**fields), self.user_id, self.product_id
        )

    async def track_row(self, track_id: UUID) -> StudioTrack:
        return (
            await self.session.execute(
                select(StudioTrack)
                .where(StudioTrack.id == track_id)
                .execution_options(populate_existing=True)
            )
        ).scalar_one()

    async def version_count(self, track_id: UUID | None = None) -> int:
        query = select(func.count()).select_from(StudioDocumentVersion)
        if track_id is not None:
            query = query.where(StudioDocumentVersion.track_id == track_id)
        return (await self.session.execute(query)).scalar_one()

    async def version_row(self, version_id: UUID) -> StudioDocumentVersion:
        return (
            await self.session.execute(
                select(StudioDocumentVersion).where(StudioDocumentVersion.id == version_id)
            )
        ).scalar_one()
