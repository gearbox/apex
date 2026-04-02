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
import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import User
from src.db.session import DatabaseManager

logger = structlog.get_logger(__name__)
console = Console()

app = typer.Typer(
    name="manage-admin",
    help="Admin role management CLI — break-glass recovery tool",
    no_args_is_help=True,
)

PRODUCT_OPTION = typer.Option(..., "--product", "-p", help="Product slug (vex, synthara)")


async def _get_session() -> tuple[DatabaseManager, AsyncSession]:
    """Create a database session from settings."""
    from src.core.config import Settings

    settings = Settings()
    db = DatabaseManager(settings.database_url)
    session = db.session_factory()
    return db, session


def with_session(fn: Callable[..., Awaitable[None]]) -> Callable[..., None]:
    """Decorator that wraps an async CLI command with DB session lifecycle.

    The decorated function receives `session` as its first argument.
    Session cleanup and DB shutdown are handled automatically.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        async def _run() -> None:
            db, session = await _get_session()
            try:
                await fn(session, *args, **kwargs)
            finally:
                await session.close()
                await db.close()

        asyncio.run(_run())

    # Remove 'session' from the typer-visible signature so it isn't treated as a CLI arg
    sig = inspect.signature(fn)
    params = [p for name, p in sig.parameters.items() if name != "session"]
    wrapper.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
    return wrapper


async def _resolve_user_or_exit(session: AsyncSession, email: str, product_id: str) -> User:
    """Find an active user by email and product, or exit with error."""
    result = await session.execute(
        select(User).where(
            User.email == email,
            User.product_id == product_id,
            User.is_active == True,  # noqa: E712
        )
    )
    user = result.scalar_one_or_none()
    if user is None:
        console.print(f"[red]User '{email}' not found in product '{product_id}'[/red]")
        raise typer.Exit(1)
    return user


@app.command("grant-superadmin")
@with_session
async def grant_superadmin(
    session: AsyncSession,
    email: str = typer.Argument(..., help="User email"),
    product: str = PRODUCT_OPTION,
) -> None:
    """Grant SUPERADMIN role to a user. This is the primary bootstrap command."""
    from src.api.services.admin_management import AdminManagementService
    from src.core.constants import SYSTEM_USER_ID
    from src.core.enums import UserRole

    user = await _resolve_user_or_exit(session, email, product)
    service = AdminManagementService()

    try:
        await service.grant_role(
            actor_id=SYSTEM_USER_ID,
            target_user_id=user.id,
            new_role=UserRole.SUPERADMIN,
            product_id=product,
            source="cli",
            session=session,
        )
        await session.commit()
        console.print(
            f"[green]✓[/green] Granted SUPERADMIN to {email} "
            f"(product: {product}, user_id: {user.id})"
        )
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command("revoke-superadmin")
@with_session
async def revoke_superadmin(
    session: AsyncSession,
    email: str = typer.Argument(..., help="User email"),
    product: str = PRODUCT_OPTION,
    force: bool = typer.Option(False, "--force", "-f", help="Bypass last-superadmin check"),
) -> None:
    """Revoke SUPERADMIN role, demoting to USER. Blocked if last superadmin (unless --force)."""
    from src.api.services.admin_management import AdminManagementService, LastSuperadminError
    from src.core.constants import SYSTEM_USER_ID

    user = await _resolve_user_or_exit(session, email, product)
    service = AdminManagementService()

    try:
        await service.revoke_role(
            actor_id=SYSTEM_USER_ID,
            target_user_id=user.id,
            product_id=product,
            source="cli",
            session=session,
        )
        await session.commit()
        console.print(f"[green]✓[/green] Revoked SUPERADMIN from {email} (product: {product})")
    except LastSuperadminError as exc:
        if force:
            from sqlalchemy import delete as sa_delete

            from src.core.enums import UserRole
            from src.core.uid import new_id
            from src.db.models.admin import AdminAuditLog, AdminPermissionGrant

            user.role = UserRole.USER
            await session.execute(
                sa_delete(AdminPermissionGrant).where(
                    AdminPermissionGrant.user_id == user.id,
                    AdminPermissionGrant.product_id == product,
                )
            )
            session.add(
                AdminAuditLog(
                    id=new_id(),
                    actor_id=SYSTEM_USER_ID,
                    target_user_id=user.id,
                    product_id=product,
                    action="role.revoke",
                    detail="CLI --force: revoked last superadmin",
                    source="cli",
                )
            )
            await session.commit()
            console.print(
                f"[yellow]⚠[/yellow] Force-revoked last SUPERADMIN from {email} "
                f"(product: {product})"
            )
        else:
            console.print(f"[red]{exc}[/red]")
            console.print("[dim]Use --force to override.[/dim]")
            raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command("grant-admin")
@with_session
async def grant_admin(
    session: AsyncSession,
    email: str = typer.Argument(..., help="User email"),
    product: str = PRODUCT_OPTION,
) -> None:
    """Grant ADMIN role to a user."""
    from src.api.services.admin_management import AdminManagementService
    from src.core.constants import SYSTEM_USER_ID
    from src.core.enums import UserRole

    user = await _resolve_user_or_exit(session, email, product)
    service = AdminManagementService()

    try:
        await service.grant_role(
            actor_id=SYSTEM_USER_ID,
            target_user_id=user.id,
            new_role=UserRole.ADMIN,
            product_id=product,
            source="cli",
            session=session,
        )
        await session.commit()
        console.print(f"[green]✓[/green] Granted ADMIN to {email} (product: {product})")
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command("revoke-admin")
@with_session
async def revoke_admin(
    session: AsyncSession,
    email: str = typer.Argument(..., help="User email"),
    product: str = PRODUCT_OPTION,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Bypass last-superadmin check (if target is superadmin)",
    ),
) -> None:
    """Revoke admin role, demoting to USER. Also revokes all permissions.

    If the target is a superadmin, the same last-superadmin guard applies.
    Use --force to bypass.
    """
    from src.api.services.admin_management import AdminManagementService, LastSuperadminError
    from src.core.constants import SYSTEM_USER_ID

    user = await _resolve_user_or_exit(session, email, product)
    service = AdminManagementService()

    try:
        await service.revoke_role(
            actor_id=SYSTEM_USER_ID,
            target_user_id=user.id,
            product_id=product,
            source="cli",
            session=session,
        )
        await session.commit()
        console.print(f"[green]✓[/green] Revoked admin from {email} (product: {product})")
    except LastSuperadminError as exc:
        if force:
            from sqlalchemy import delete as sa_delete

            from src.core.enums import UserRole
            from src.core.uid import new_id
            from src.db.models.admin import AdminAuditLog, AdminPermissionGrant

            user.role = UserRole.USER
            await session.execute(
                sa_delete(AdminPermissionGrant).where(
                    AdminPermissionGrant.user_id == user.id,
                    AdminPermissionGrant.product_id == product,
                )
            )
            session.add(
                AdminAuditLog(
                    id=new_id(),
                    actor_id=SYSTEM_USER_ID,
                    target_user_id=user.id,
                    product_id=product,
                    action="role.revoke",
                    detail="CLI --force: revoked last superadmin via revoke-admin",
                    source="cli",
                )
            )
            await session.commit()
            console.print(f"[yellow]⚠[/yellow] Force-revoked from {email} (product: {product})")
        else:
            console.print(f"[red]{exc}[/red]")
            console.print("[dim]Use --force to override.[/dim]")
            raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command("grant-permission")
@with_session
async def grant_permission(
    session: AsyncSession,
    email: str = typer.Argument(..., help="User email"),
    permission: str = typer.Argument(..., help="Permission key (e.g. billing_adjust)"),
    product: str = PRODUCT_OPTION,
) -> None:
    """Grant a permission to an admin user."""
    from src.api.services.admin_management import AdminManagementService
    from src.core.constants import SYSTEM_USER_ID
    from src.core.enums import AdminPermission

    try:
        perm = AdminPermission(permission)
    except ValueError:
        valid = ", ".join(p.value for p in AdminPermission)
        console.print(f"[red]Invalid permission '{permission}'. Valid: {valid}[/red]")
        raise typer.Exit(1) from None

    user = await _resolve_user_or_exit(session, email, product)
    service = AdminManagementService()

    try:
        await service.grant_permission(
            actor_id=SYSTEM_USER_ID,
            target_user_id=user.id,
            permission=perm,
            product_id=product,
            source="cli",
            session=session,
        )
        await session.commit()
        console.print(f"[green]✓[/green] Granted '{perm.value}' to {email} (product: {product})")
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command("revoke-permission")
@with_session
async def revoke_permission(
    session: AsyncSession,
    email: str = typer.Argument(..., help="User email"),
    permission: str = typer.Argument(..., help="Permission key (e.g. billing_adjust)"),
    product: str = PRODUCT_OPTION,
) -> None:
    """Revoke a permission from a user."""
    from src.api.services.admin_management import AdminManagementService
    from src.core.constants import SYSTEM_USER_ID
    from src.core.enums import AdminPermission

    try:
        perm = AdminPermission(permission)
    except ValueError:
        valid = ", ".join(p.value for p in AdminPermission)
        console.print(f"[red]Invalid permission '{permission}'. Valid: {valid}[/red]")
        raise typer.Exit(1) from None

    user = await _resolve_user_or_exit(session, email, product)
    service = AdminManagementService()

    try:
        await service.revoke_permission(
            actor_id=SYSTEM_USER_ID,
            target_user_id=user.id,
            permission=perm,
            product_id=product,
            source="cli",
            session=session,
        )
        await session.commit()
        console.print(f"[green]✓[/green] Revoked '{perm.value}' from {email} (product: {product})")
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command("list-admins")
@with_session
async def list_admins(
    session: AsyncSession,
    product: str = PRODUCT_OPTION,
) -> None:
    """List all admin and superadmin users for a product."""
    from src.db.repositories.admin import AdminRepository
    from src.db.repositories.user import UserRepository

    user_repo = UserRepository(session)
    users = await user_repo.list_users_by_roles(
        product_id=product,
        roles=["admin", "superadmin"],
    )

    if not users:
        console.print(f"[yellow]No admins found for product '{product}'[/yellow]")
        return

    admin_repo = AdminRepository(session)
    permissions_map = await admin_repo.get_permissions_batch([u.id for u in users], product)

    table = Table(title=f"Admins — {product}")
    table.add_column("Email", style="cyan")
    table.add_column("Role", style="bold")
    table.add_column("Permissions", style="green")
    table.add_column("User ID", style="dim")

    for u in users:
        perms = permissions_map.get(u.id, [])
        role_str = u.role if isinstance(u.role, str) else u.role.value
        table.add_row(
            u.email,
            role_str.upper(),
            ", ".join(perms) if perms else "—",
            str(u.id),
        )

    console.print(table)


@app.command("audit-log")
@with_session
async def audit_log(
    session: AsyncSession,
    product: str = PRODUCT_OPTION,
    limit: int = typer.Option(20, "--limit", "-n", help="Number of entries"),
) -> None:
    """Show recent audit log entries."""
    from src.db.repositories.admin import AdminRepository

    admin_repo = AdminRepository(session)
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


if __name__ == "__main__":
    app()
