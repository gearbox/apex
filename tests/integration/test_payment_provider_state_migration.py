"""Round-trip the payment provider state migration."""

from alembic.config import Config

from alembic import command


def test_revision_017_downgrade_upgrade_round_trip(
    test_database_url: str,
    db_engine,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    assert db_engine is not None
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("DEBUG", "false")
    config = Config("alembic.ini")
    try:
        command.downgrade(config, "016")
    finally:
        command.upgrade(config, "head")
