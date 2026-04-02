"""CLI tool for admin role management — break-glass recovery.

Connects directly to the database, bypassing the API entirely.
Uses SYSTEM_USER_ID as the actor for all CLI-initiated changes.

Usage:
    python -m src.cli.manage_admin grant-superadmin user@example.com --product vex
    python -m src.cli.manage_admin revoke-superadmin user@example.com --product vex
    python -m src.cli.manage_admin grant-admin user@example.com --product vex
    python -m src.cli.manage_admin revoke-admin user@example.com --product vex
    python -m src.cli.manage_admin list-admins --product vex
    python -m src.cli.manage_admin grant-permission user@example.com billing_adjust --product vex
    python -m src.cli.manage_admin revoke-permission user@example.com billing_adjust --product vex
    python -m src.cli.manage_admin audit-log --product vex
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import structlog
import typer
from rich.console import Console
from rich.table import Table

logger = structlog.get_logger(__name__)
console = Console()

app = typer.Typer(
    name="manage-admin",
    help="Admin role management CLI — break-glass recovery tool",
    no_args_is_help=True,
)


async def _get_session() -> tuple[object, object]:
    """Create a database session from settings."""
    from src.core.config import Settings
    from src.db.session import DatabaseManager

    settings = Settings()
    db = DatabaseManager(settings.database_url)
    await db.init()
    session = db.session_factory()
    return db, session


async def _resolve_user(session: object, email: str, product_id: str) -> object:
    """Find an active user by email and product."""
    from sqlalchemy import select

    from src.db.models import User

    result = await session.execute(  # type: ignore[union-attr]
        select(User).where(
            User.email == email,
            User.product_id == product_id,
            User.is_active == True,  # noqa: E712
        )
    )
    return result.scalar_one_or_none()


async def _write_audit(
    session: object,
    *,
    actor_id: UUID,
    target_user_id: UUID,
    product_id: str,
    action: str,
    detail: str,
) -> None:
    """Write a CLI audit entry."""
    from src.core.uid import new_id
    from src.db.models.admin import AdminAuditLog

    entry = AdminAuditLog(
        id=new_id(),
        actor_id=actor_id,
        target_user_id=target_user_id,
        product_id=product_id,
        action=action,
        detail=detail,
        source="cli",
    )
    session.add(entry)  # type: ignore[union-attr]


PRODUCT_OPTION = typer.Option(..., "--product", "-p", help="Product slug (vex, synthara)")


@app.command("grant-superadmin")
def grant_superadmin(
    email: str = typer.Argument(..., help="User email"),
    product: str = PRODUCT_OPTION,
) -> None:
    """Grant SUPERADMIN role to a user. Primary bootstrap command."""

    async def _run() -> None:
        from src.core.constants import SYSTEM_USER_ID

        db, session = await _get_session()
        try:
            user = await _resolve_user(session, email, product)
            if user is None:
                console.print(f"[red]User '{email}' not found in product '{product}'[/red]")
                raise typer.Exit(1)

            old_role = user.role if isinstance(user.role, str) else user.role.value  # type: ignore[union-attr]
            user.role = "superadmin"  # type: ignore[union-attr]
            await _write_audit(
                session,
                actor_id=SYSTEM_USER_ID,
                target_user_id=user.id,  # type: ignore[union-attr]
                product_id=product,
                action="role.grant",
                detail=f"CLI: role changed from '{old_role}' to 'superadmin'",
            )
            await session.commit()  # type: ignore[union-attr]
            console.print(
                f"[green]✓[/green] Granted SUPERADMIN to {email} "
                f"(product: {product}, user_id: {user.id})"  # type: ignore[union-attr]
            )
        finally:
            await session.close()  # type: ignore[union-attr]
            await db.shutdown()  # type: ignore[union-attr]

    asyncio.run(_run())


@app.command("revoke-superadmin")
def revoke_superadmin(
    email: str = typer.Argument(..., help="User email"),
    product: str = PRODUCT_OPTION,
    force: bool = typer.Option(False, "--force", "-f", help="Bypass last-superadmin check"),
) -> None:
    """Revoke SUPERADMIN role, demoting to USER. Blocked if last superadmin (unless --force)."""

    async def _run() -> None:
        from src.core.constants import SYSTEM_USER_ID

        db, session = await _get_session()
        try:
            user = await _resolve_user(session, email, product)
            if user is None:
                console.print(f"[red]User '{email}' not found in product '{product}'[/red]")
                raise typer.Exit(1)

            current_role = user.role if isinstance(user.role, str) else user.role.value  # type: ignore[union-attr]
            if current_role != "superadmin":
                console.print(
                    f"[yellow]User '{email}' is not a superadmin (role: {current_role})[/yellow]"
                )
                raise typer.Exit(1)

            if not force:
                from sqlalchemy import func, select

                from src.db.models import User as UserModel

                count_result = await session.execute(  # type: ignore[union-attr]
                    select(func.count(UserModel.id)).where(
                        UserModel.product_id == product,
                        UserModel.role == "superadmin",
                        UserModel.is_active == True,  # noqa: E712
                    )
                )
                if count_result.scalar_one() <= 1:
                    console.print(
                        "[red]Cannot revoke the last superadmin. Use --force to override.[/red]"
                    )
                    raise typer.Exit(1)

            user.role = "user"  # type: ignore[union-attr]

            from sqlalchemy import delete

            from src.db.models.admin import AdminPermissionGrant

            await session.execute(  # type: ignore[union-attr]
                delete(AdminPermissionGrant).where(
                    AdminPermissionGrant.user_id == user.id,  # type: ignore[union-attr]
                    AdminPermissionGrant.product_id == product,
                )
            )

            await _write_audit(
                session,
                actor_id=SYSTEM_USER_ID,
                target_user_id=user.id,  # type: ignore[union-attr]
                product_id=product,
                action="role.revoke",
                detail="CLI: role changed from 'superadmin' to 'user'",
            )
            await session.commit()  # type: ignore[union-attr]
            console.print(f"[green]✓[/green] Revoked SUPERADMIN from {email} (product: {product})")
        finally:
            await session.close()  # type: ignore[union-attr]
            await db.shutdown()  # type: ignore[union-attr]

    asyncio.run(_run())


@app.command("grant-admin")
def grant_admin(
    email: str = typer.Argument(..., help="User email"),
    product: str = PRODUCT_OPTION,
) -> None:
    """Grant ADMIN role to a user."""

    async def _run() -> None:
        from src.core.constants import SYSTEM_USER_ID

        db, session = await _get_session()
        try:
            user = await _resolve_user(session, email, product)
            if user is None:
                console.print(f"[red]User '{email}' not found in product '{product}'[/red]")
                raise typer.Exit(1)

            old_role = user.role if isinstance(user.role, str) else user.role.value  # type: ignore[union-attr]
            user.role = "admin"  # type: ignore[union-attr]
            await _write_audit(
                session,
                actor_id=SYSTEM_USER_ID,
                target_user_id=user.id,  # type: ignore[union-attr]
                product_id=product,
                action="role.grant",
                detail=f"CLI: role changed from '{old_role}' to 'admin'",
            )
            await session.commit()  # type: ignore[union-attr]
            console.print(f"[green]✓[/green] Granted ADMIN to {email} (product: {product})")
        finally:
            await session.close()  # type: ignore[union-attr]
            await db.shutdown()  # type: ignore[union-attr]

    asyncio.run(_run())


@app.command("revoke-admin")
def revoke_admin(
    email: str = typer.Argument(..., help="User email"),
    product: str = PRODUCT_OPTION,
) -> None:
    """Revoke ADMIN role, demoting to USER. Also revokes all permissions."""

    async def _run() -> None:
        from sqlalchemy import delete

        from src.core.constants import SYSTEM_USER_ID
        from src.db.models.admin import AdminPermissionGrant

        db, session = await _get_session()
        try:
            user = await _resolve_user(session, email, product)
            if user is None:
                console.print(f"[red]User '{email}' not found in product '{product}'[/red]")
                raise typer.Exit(1)

            current_role = user.role if isinstance(user.role, str) else user.role.value  # type: ignore[union-attr]
            if current_role not in ("admin", "superadmin"):
                console.print(
                    f"[yellow]User '{email}' is not an admin (role: {current_role})[/yellow]"
                )
                raise typer.Exit(1)

            user.role = "user"  # type: ignore[union-attr]
            await session.execute(  # type: ignore[union-attr]
                delete(AdminPermissionGrant).where(
                    AdminPermissionGrant.user_id == user.id,  # type: ignore[union-attr]
                    AdminPermissionGrant.product_id == product,
                )
            )
            await _write_audit(
                session,
                actor_id=SYSTEM_USER_ID,
                target_user_id=user.id,  # type: ignore[union-attr]
                product_id=product,
                action="role.revoke",
                detail=f"CLI: role changed from '{current_role}' to 'user'",
            )
            await session.commit()  # type: ignore[union-attr]
            console.print(f"[green]✓[/green] Revoked admin from {email} (product: {product})")
        finally:
            await session.close()  # type: ignore[union-attr]
            await db.shutdown()  # type: ignore[union-attr]

    asyncio.run(_run())


@app.command("list-admins")
def list_admins(
    product: str = PRODUCT_OPTION,
) -> None:
    """List all admin and superadmin users for a product."""

    async def _run() -> None:
        from sqlalchemy import select

        from src.db.models import User as UserModel
        from src.db.models.admin import AdminPermissionGrant

        db, session = await _get_session()
        try:
            result = await session.execute(  # type: ignore[union-attr]
                select(UserModel).where(
                    UserModel.product_id == product,
                    UserModel.role.in_(["admin", "superadmin"]),
                    UserModel.is_active == True,  # noqa: E712
                )
            )
            users = result.scalars().all()

            if not users:
                console.print(f"[yellow]No admins found for product '{product}'[/yellow]")
                return

            table = Table(title=f"Admins — {product}")
            table.add_column("Email", style="cyan")
            table.add_column("Role", style="bold")
            table.add_column("Permissions", style="green")
            table.add_column("User ID", style="dim")

            for u in users:
                perm_result = await session.execute(  # type: ignore[union-attr]
                    select(AdminPermissionGrant.permission).where(
                        AdminPermissionGrant.user_id == u.id,
                        AdminPermissionGrant.product_id == product,
                    )
                )
                perms = [p for (p,) in perm_result.all()]
                role_str = u.role if isinstance(u.role, str) else u.role.value
                table.add_row(
                    u.email,
                    role_str.upper(),
                    ", ".join(perms) if perms else "—",
                    str(u.id),
                )

            console.print(table)
        finally:
            await session.close()  # type: ignore[union-attr]
            await db.shutdown()  # type: ignore[union-attr]

    asyncio.run(_run())


@app.command("grant-permission")
def grant_permission(
    email: str = typer.Argument(..., help="User email"),
    permission: str = typer.Argument(..., help="Permission key (e.g. billing_adjust)"),
    product: str = PRODUCT_OPTION,
) -> None:
    """Grant a permission to an admin user."""

    async def _run() -> None:
        from src.core.enums import AdminPermission

        try:
            perm = AdminPermission(permission)
        except ValueError:
            valid = ", ".join(p.value for p in AdminPermission)
            console.print(f"[red]Invalid permission '{permission}'. Valid: {valid}[/red]")
            raise typer.Exit(1) from None

        from src.core.constants import SYSTEM_USER_ID
        from src.core.uid import new_id
        from src.db.models.admin import AdminPermissionGrant
        from src.db.repositories.admin import AdminRepository

        db, session = await _get_session()
        try:
            user = await _resolve_user(session, email, product)
            if user is None:
                console.print(f"[red]User '{email}' not found in product '{product}'[/red]")
                raise typer.Exit(1)

            admin_repo = AdminRepository(session)  # type: ignore[arg-type]
            already = await admin_repo.has_permission(user.id, perm.value, product)  # type: ignore[union-attr]
            if already:
                console.print(f"[yellow]User already has '{perm.value}' permission[/yellow]")
                return

            session.add(  # type: ignore[union-attr]
                AdminPermissionGrant(
                    id=new_id(),
                    user_id=user.id,  # type: ignore[union-attr]
                    permission=perm.value,
                    product_id=product,
                    granted_by=SYSTEM_USER_ID,
                )
            )
            await _write_audit(
                session,
                actor_id=SYSTEM_USER_ID,
                target_user_id=user.id,  # type: ignore[union-attr]
                product_id=product,
                action="permission.grant",
                detail=f"CLI: granted permission '{perm.value}'",
            )
            await session.commit()  # type: ignore[union-attr]
            console.print(
                f"[green]✓[/green] Granted '{perm.value}' to {email} (product: {product})"
            )
        finally:
            await session.close()  # type: ignore[union-attr]
            await db.shutdown()  # type: ignore[union-attr]

    asyncio.run(_run())


@app.command("revoke-permission")
def revoke_permission(
    email: str = typer.Argument(..., help="User email"),
    permission: str = typer.Argument(..., help="Permission key (e.g. billing_adjust)"),
    product: str = PRODUCT_OPTION,
) -> None:
    """Revoke a permission from a user."""

    async def _run() -> None:
        from src.core.enums import AdminPermission

        try:
            perm = AdminPermission(permission)
        except ValueError:
            valid = ", ".join(p.value for p in AdminPermission)
            console.print(f"[red]Invalid permission '{permission}'. Valid: {valid}[/red]")
            raise typer.Exit(1) from None

        from src.core.constants import SYSTEM_USER_ID
        from src.db.repositories.admin import AdminRepository

        db, session = await _get_session()
        try:
            user = await _resolve_user(session, email, product)
            if user is None:
                console.print(f"[red]User '{email}' not found in product '{product}'[/red]")
                raise typer.Exit(1)

            admin_repo = AdminRepository(session)  # type: ignore[arg-type]
            deleted = await admin_repo.revoke_permission(user.id, perm.value, product)  # type: ignore[union-attr]
            if not deleted:
                console.print(f"[yellow]User doesn't have '{perm.value}' permission[/yellow]")
                return

            await _write_audit(
                session,
                actor_id=SYSTEM_USER_ID,
                target_user_id=user.id,  # type: ignore[union-attr]
                product_id=product,
                action="permission.revoke",
                detail=f"CLI: revoked permission '{perm.value}'",
            )
            await session.commit()  # type: ignore[union-attr]
            console.print(
                f"[green]✓[/green] Revoked '{perm.value}' from {email} (product: {product})"
            )
        finally:
            await session.close()  # type: ignore[union-attr]
            await db.shutdown()  # type: ignore[union-attr]

    asyncio.run(_run())


@app.command("audit-log")
def audit_log(
    product: str = PRODUCT_OPTION,
    limit: int = typer.Option(20, "--limit", "-n", help="Number of entries"),
) -> None:
    """Show recent audit log entries."""

    async def _run() -> None:
        from src.db.repositories.admin import AdminRepository

        db, session = await _get_session()
        try:
            admin_repo = AdminRepository(session)  # type: ignore[arg-type]
            entries = await admin_repo.get_audit_log(product, limit=limit)

            if not entries:
                console.print(f"[yellow]No audit entries for product '{product}'[/yellow]")
                return

            table = Table(title=f"Admin Audit Log — {product}")
            table.add_column("Time", style="dim")
            table.add_column("Action", style="bold")
            table.add_column("Actor", style="cyan")
            table.add_column("Target", style="yellow")
            table.add_column("Detail")
            table.add_column("Source", style="dim")

            for e in entries:
                table.add_row(
                    str(e.created_at)[:19],
                    e.action,
                    str(e.actor_id)[:8] + "…",
                    str(e.target_user_id)[:8] + "…",
                    e.detail,
                    e.source,
                )

            console.print(table)
        finally:
            await session.close()  # type: ignore[union-attr]
            await db.shutdown()  # type: ignore[union-attr]

    asyncio.run(_run())


if __name__ == "__main__":
    app()
