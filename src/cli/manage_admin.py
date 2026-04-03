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
from collections.abc import Awaitable, Callable

import structlog
import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.services.admin_management import AdminManagementService, LastSuperadminError
from src.core.config import Settings
from src.core.constants import SYSTEM_USER_ID
from src.core.enums import AdminPermission, UserRole
from src.db.models import User
from src.db.repositories.admin import AdminRepository
from src.db.repositories.user import UserRepository
from src.db.session import DatabaseManager

logger = structlog.get_logger(__name__)
console = Console()

app = typer.Typer(
    name="manage-admin",
    help="Admin role management CLI — break-glass recovery tool",
    no_args_is_help=True,
)

PRODUCT_OPTION = typer.Option(..., "--product", "-p", help="Product slug (vex, synthara)")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


async def _get_session() -> tuple[DatabaseManager, AsyncSession]:
    """Create a database session from settings."""
    settings = Settings()
    db = DatabaseManager(settings.database_url)
    session = db.session_factory()
    return db, session


async def _run_with_session(fn: Callable[[AsyncSession], Awaitable[None]]) -> None:
    """Run an async function with a managed database session."""
    db, session = await _get_session()
    try:
        await fn(session)
    finally:
        await session.close()
        await db.close()


def _run_cli(handler: Callable[[AsyncSession], Awaitable[None]]) -> None:
    """Wire a Typer command to the async session lifecycle."""
    asyncio.run(_run_with_session(handler))


async def _resolve_user(session: AsyncSession, email: str, product_id: str) -> User | None:
    """Find an active user by email and product. Pure data access."""
    result = await session.execute(
        select(User).where(
            User.email == email,
            User.product_id == product_id,
            User.is_active == True,  # noqa: E712
        )
    )
    return result.scalar_one_or_none()


async def _resolve_user_or_exit(session: AsyncSession, email: str, product_id: str) -> User:
    """Find an active user by email and product, or exit with error."""
    user = await _resolve_user(session, email, product_id)
    if user is None:
        console.print(f"[red]User '{email}' not found in product '{product_id}'[/red]")
        raise typer.Exit(1)
    return user


async def _run_or_exit(
    session: AsyncSession,
    coro: Awaitable[None],
    success_message: str,
) -> None:
    """Execute a coroutine, commit on success, print error and exit on failure."""
    try:
        await coro
        await session.commit()
        console.print(success_message)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


async def _with_user_and_service(
    session: AsyncSession,
    email: str,
    product: str,
    fn: Callable[..., Awaitable[None]],
    success_message: str,
) -> None:
    """Resolve user, create service, call fn(service, user), commit, print."""
    user = await _resolve_user_or_exit(session, email, product)
    service = AdminManagementService()
    await _run_or_exit(session, fn(service, user), success_message)


async def _revoke_with_force_handling(
    session: AsyncSession,
    email: str,
    product: str,
    force: bool,
    *,
    normal_msg: str,
    force_msg: str,
) -> None:
    """Revoke an admin/superadmin role with shared LastSuperadminError handling."""
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
        console.print(normal_msg)
    except LastSuperadminError as exc:
        if force:
            await service.force_revoke_role(
                actor_id=SYSTEM_USER_ID,
                target_user_id=user.id,
                product_id=product,
                source="cli",
                session=session,
            )
            await session.commit()
            console.print(force_msg)
        else:
            console.print(f"[red]{exc}[/red]")
            console.print("[dim]Use --force to override.[/dim]")
            raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def _grant_superadmin_impl(session: AsyncSession, email: str, product: str) -> None:
    async def op(service: object, user: User) -> None:
        assert isinstance(service, AdminManagementService)
        await service.grant_role(
            actor_id=SYSTEM_USER_ID,
            target_user_id=user.id,
            new_role=UserRole.SUPERADMIN,
            product_id=product,
            source="cli",
            session=session,
        )

    # Resolve once via _with_user_and_service; include user_id in message
    # by deferring message construction.
    user = await _resolve_user_or_exit(session, email, product)
    service = AdminManagementService()
    await _run_or_exit(
        session,
        op(service, user),
        f"[green]✓[/green] Granted SUPERADMIN to {email} (product: {product}, user_id: {user.id})",
    )


@app.command("grant-superadmin")
def grant_superadmin(
    email: str = typer.Argument(..., help="User email"),
    product: str = PRODUCT_OPTION,
) -> None:
    """Grant SUPERADMIN role to a user. This is the primary bootstrap command."""
    _run_cli(lambda s: _grant_superadmin_impl(s, email, product))


async def _grant_admin_impl(session: AsyncSession, email: str, product: str) -> None:
    async def op(service: object, user: User) -> None:
        assert isinstance(service, AdminManagementService)
        await service.grant_role(
            actor_id=SYSTEM_USER_ID,
            target_user_id=user.id,
            new_role=UserRole.ADMIN,
            product_id=product,
            source="cli",
            session=session,
        )

    await _with_user_and_service(
        session,
        email,
        product,
        op,
        f"[green]✓[/green] Granted ADMIN to {email} (product: {product})",
    )


@app.command("grant-admin")
def grant_admin(
    email: str = typer.Argument(..., help="User email"),
    product: str = PRODUCT_OPTION,
) -> None:
    """Grant ADMIN role to a user."""
    _run_cli(lambda s: _grant_admin_impl(s, email, product))


@app.command("revoke-superadmin")
def revoke_superadmin(
    email: str = typer.Argument(..., help="User email"),
    product: str = PRODUCT_OPTION,
    force: bool = typer.Option(False, "--force", "-f", help="Bypass last-superadmin check"),
) -> None:
    """Revoke SUPERADMIN role, demoting to USER. Blocked if last superadmin (unless --force)."""
    _run_cli(
        lambda s: _revoke_with_force_handling(
            s,
            email,
            product,
            force,
            normal_msg=f"[green]✓[/green] Revoked SUPERADMIN from {email} (product: {product})",
            force_msg=(
                f"[yellow]⚠[/yellow] Force-revoked last SUPERADMIN from {email} "
                f"(product: {product})"
            ),
        )
    )


@app.command("revoke-admin")
def revoke_admin(
    email: str = typer.Argument(..., help="User email"),
    product: str = PRODUCT_OPTION,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Bypass last-superadmin check (if target is superadmin)",
    ),
) -> None:
    """Revoke admin role, demoting to USER. Also revokes all permissions."""
    _run_cli(
        lambda s: _revoke_with_force_handling(
            s,
            email,
            product,
            force,
            normal_msg=f"[green]✓[/green] Revoked admin from {email} (product: {product})",
            force_msg=f"[yellow]⚠[/yellow] Force-revoked from {email} (product: {product})",
        )
    )


async def _grant_permission_impl(
    session: AsyncSession, email: str, permission: str, product: str
) -> None:
    try:
        perm = AdminPermission(permission)
    except ValueError:
        valid = ", ".join(p.value for p in AdminPermission)
        console.print(f"[red]Invalid permission '{permission}'. Valid: {valid}[/red]")
        raise typer.Exit(1) from None

    async def op(service: object, user: User) -> None:
        assert isinstance(service, AdminManagementService)
        await service.grant_permission(
            actor_id=SYSTEM_USER_ID,
            target_user_id=user.id,
            permission=perm,
            product_id=product,
            source="cli",
            session=session,
        )

    await _with_user_and_service(
        session,
        email,
        product,
        op,
        f"[green]✓[/green] Granted '{perm.value}' to {email} (product: {product})",
    )


@app.command("grant-permission")
def grant_permission(
    email: str = typer.Argument(..., help="User email"),
    permission: str = typer.Argument(..., help="Permission key (e.g. billing_adjust)"),
    product: str = PRODUCT_OPTION,
) -> None:
    """Grant a permission to an admin user."""
    _run_cli(lambda s: _grant_permission_impl(s, email, permission, product))


async def _revoke_permission_impl(
    session: AsyncSession, email: str, permission: str, product: str
) -> None:
    try:
        perm = AdminPermission(permission)
    except ValueError:
        valid = ", ".join(p.value for p in AdminPermission)
        console.print(f"[red]Invalid permission '{permission}'. Valid: {valid}[/red]")
        raise typer.Exit(1) from None

    async def op(service: object, user: User) -> None:
        assert isinstance(service, AdminManagementService)
        await service.revoke_permission(
            actor_id=SYSTEM_USER_ID,
            target_user_id=user.id,
            permission=perm,
            product_id=product,
            source="cli",
            session=session,
        )

    await _with_user_and_service(
        session,
        email,
        product,
        op,
        f"[green]✓[/green] Revoked '{perm.value}' from {email} (product: {product})",
    )


@app.command("revoke-permission")
def revoke_permission(
    email: str = typer.Argument(..., help="User email"),
    permission: str = typer.Argument(..., help="Permission key (e.g. billing_adjust)"),
    product: str = PRODUCT_OPTION,
) -> None:
    """Revoke a permission from a user."""
    _run_cli(lambda s: _revoke_permission_impl(s, email, permission, product))


async def _list_admins_impl(session: AsyncSession, product: str) -> None:
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


@app.command("list-admins")
def list_admins(product: str = PRODUCT_OPTION) -> None:
    """List all admin and superadmin users for a product."""
    _run_cli(lambda s: _list_admins_impl(s, product))


async def _audit_log_impl(session: AsyncSession, product: str, limit: int) -> None:
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
            f"{str(e.actor_id)[:8]}…",
            f"{str(e.target_user_id)[:8]}…",
            e.detail,
            e.source,
        )

    console.print(table)


@app.command("audit-log")
def audit_log(
    product: str = PRODUCT_OPTION,
    limit: int = typer.Option(20, "--limit", "-n", help="Number of entries"),
) -> None:
    """Show recent audit log entries."""
    _run_cli(lambda s: _audit_log_impl(s, product, limit))


if __name__ == "__main__":
    app()
