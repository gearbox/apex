"""Studio lyrics runs: what a run works on (``studio_runs.inputs``).

Revision ID: 058
Revises: 057
Create Date: 2026-11-15 00:00:00.000000

One nullable column and two CHECK constraints. ``inputs`` records what a run works on - the plan
(and, for a revision, the lyrics) and the language - resolved **when the run was accepted**
(P2a-40), so what the run does never depends on where the track's current pointers are when a
worker claims it. It is SQL ``NULL`` for ``plan`` runs (and for every run that existed before this
revision); the model writes ``None`` as SQL ``NULL`` (``JSONB(none_as_null=True)``), never as the
JSON value ``null``.

The constraints make two rules structural instead of conventional:

* *inputs by kind* - ``inputs IS NULL`` exactly for ``plan`` runs;
* *result by kind* - a ``plan`` run has no lyrics result, and only a ``plan`` run has a plan result.

There is deliberately no foreign key on ``inputs``: ``ON DELETE`` would either block deleting a
version for good or erase the record. "A version an active run takes as input cannot be deleted" is
a service rule (``409 version_in_use_by_run``), checked under the track lock.

``error_detail`` (added by 057, and stored as JSON ``null`` for a failure with no detail) is cleaned
the same way: the upgrade turns every JSON ``null`` into SQL ``NULL`` (a downgrade leaves them so).

Downgrade drops the constraints and the column (and the record of what each run worked on). Once
``inputs`` is gone a ``lyrics`` / ``revise_lyrics`` run cannot be represented in the 057 schema, and
an upgrade back would fail the constraint, so the downgrade **refuses** while any such run exists
(the same fail-loud rule as migration 056).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "058"
down_revision: str | Sequence[str] | None = "057"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INPUTS_BY_KIND = "ck_studio_runs_inputs_by_kind"
_RESULT_BY_KIND = "ck_studio_runs_result_by_kind"


def upgrade() -> None:
    """Upgrade database schema."""
    op.add_column(
        "studio_runs", sa.Column("inputs", postgresql.JSONB(none_as_null=True), nullable=True)
    )
    # A development database that ran an earlier form of this revision holds the JSON value
    # ``null`` where the rule says SQL ``NULL``. A no-op on a fresh database.
    op.execute("UPDATE studio_runs SET inputs = NULL WHERE inputs = 'null'::jsonb")
    # ``error_detail`` (migration 057) had the same flaw: a failure with no detail was stored as the
    # JSON value ``null``. The model now writes SQL ``NULL``; this cleans the rows already written.
    op.execute("UPDATE studio_runs SET error_detail = NULL WHERE error_detail = 'null'::jsonb")
    op.create_check_constraint(_INPUTS_BY_KIND, "studio_runs", "(kind = 'plan') = (inputs IS NULL)")
    op.create_check_constraint(
        _RESULT_BY_KIND,
        "studio_runs",
        "(kind = 'plan' OR result_plan_version_id IS NULL)"
        " AND (kind <> 'plan' OR result_lyrics_version_id IS NULL)",
    )


def downgrade() -> None:
    """Downgrade database schema.

    Raises:
        RuntimeError: A ``lyrics`` or ``revise_lyrics`` run exists. Nothing was changed; the 057
            schema cannot represent it, so delete those runs first.
    """
    unrepresentable = (
        op.get_bind()
        .execute(sa.text("SELECT count(*) FROM studio_runs WHERE kind <> 'plan'"))
        .scalar_one()
    )
    if unrepresentable:
        raise RuntimeError(
            f"cannot downgrade to 057: {unrepresentable} studio run(s) are lyrics or "
            "revise_lyrics runs, which the 057 schema cannot represent once `inputs` is dropped; "
            "delete those runs first"
        )
    # ``if_exists``: a development database that ran an earlier form of this revision has the
    # column but not the constraints.
    op.drop_constraint(_RESULT_BY_KIND, "studio_runs", type_="check", if_exists=True)
    op.drop_constraint(_INPUTS_BY_KIND, "studio_runs", type_="check", if_exists=True)
    op.drop_column("studio_runs", "inputs")
