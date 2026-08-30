# Pagination Unification — Remove `COUNT(*)`, Cursor-Only

> **Status:** Design complete — ready for implementation prompt  
> **Author:** Claude × Miša  
> **Date:** 2026-03-24  
> **Scope:** Project-wide pagination refactor — drop `total`/`offset`, unify on cursor + `limit+1` pattern

---

## 1. Motivation

Every paginated endpoint currently runs a `COUNT(*)` query alongside the data query. On tables that grow with usage (`generation_jobs`, `generation_outputs`, `token_transactions`), `COUNT(*)` becomes increasingly expensive — PostgreSQL must scan the entire filtered result set to produce the count, and there's no shortcut for filtered counts.

For infinite-scroll UIs, `total` is never displayed — only `has_more` and `next_cursor` matter. For admin tables, `total` provides marginal value vs. the query cost.

**Decision:** Remove `COUNT(*)` from all paginated endpoints. Use the `limit + 1` fetch pattern universally. Drop `offset` as a query parameter (cursor-only). Drop `total` from the response schema.

---

## 2. Audit: All `COUNT(*)` Usages

### 2.1 Paginated Endpoint Consumers (return `PaginatedResponse`)

| Endpoint | Route File | Service/Repo | COUNT Query | Impact |
|---|---|---|---|---|
| `GET /v1/jobs` | `routes/jobs.py` | `UnifiedJobService.list_jobs()` | `SELECT COUNT(*) FROM generation_jobs WHERE user_id = ?` | **Remove** — infinite scroll |
| `GET /v1/storage/outputs` | `routes/storage.py` | `UserContentService.list_user_outputs()` → `StorageRepository.count_user_outputs()` | `SELECT COUNT(*) FROM generation_outputs WHERE user_id = ?` | **Remove** — infinite scroll |
| `GET /v1/storage/uploads` | `routes/storage.py` | `UserContentService.list_user_uploads()` → `StorageRepository.count_user_images()` | `SELECT COUNT(*) FROM user_images WHERE user_id = ?` | **Remove** — infinite scroll |
| `GET /v1/billing/transactions` | `routes/billing.py` | `BillingService.get_transaction_history()` → `BillingRepository.get_transaction_history()` | `SELECT COUNT(*) FROM token_transactions WHERE account_id = ?` | **Remove** — infinite scroll |
| `GET /v1/storage/jobs/{id}/outputs` | `routes/storage.py` | `UserContentService.list_job_outputs()` | No COUNT (uses `len(items)`) | Already fine — bounded result set |
| `GET /v1/admin/users` | `routes/admin.py` | `UserRepository.list_users()` | `SELECT COUNT(*) FROM users WHERE ...` | **Remove** — use `limit+1` |
| `GET /v1/admin/payments` | `routes/admin.py` | `BillingRepository.list_payments()` | `SELECT COUNT(*) FROM payments WHERE ...` | **Remove** — use `limit+1` |
| `GET /v1/admin/accounts/{id}/transactions` | `routes/admin.py` | `BillingService.get_transaction_history()` | Same as billing transactions | **Remove** — shared code path |

### 2.2 Non-Pagination COUNT Usages (KEEP — different purpose)

| Location | Query | Purpose | Action |
|---|---|---|---|
| `StorageRepository.get_user_storage_stats()` | `COUNT(*)` + `SUM(size_bytes)` on uploads + outputs | Storage stats endpoint (`GET /v1/storage/stats`) | **Keep** — aggregate endpoint, not pagination |
| `UserRepository.get_user_job_count()` | `COUNT(*)` grouped by status | Admin analytics / user profile | **Keep** — aggregate, not pagination |
| `UserRepository.get_user_output_count()` | `COUNT(*)` on outputs | Admin analytics | **Keep** — aggregate, not pagination |
| `UserRepository.get_user_upload_count()` | `COUNT(*)` on uploads | Admin analytics | **Keep** — aggregate, not pagination |
| `UserRepository.count_job_outputs()` | `COUNT(*)` on outputs for a job | Used by unified job service | **Keep** — bounded by batch size (≤10) |
| `StorageRepository.delete_outputs_batch()` | `COUNT(*)` before DELETE | Counts rows to be deleted | **Keep** — operational, not pagination |

### 2.3 Repository Methods That Return `(data, total)` Tuples

These are the repository signatures that need to change:

| Repository | Method | Current Return | New Return |
|---|---|---|---|
| `StorageRepository` | `count_user_outputs()` | `int` | **Delete** — no longer called by pagination |
| `StorageRepository` | `count_user_images()` | `int` | **Delete** — no longer called by pagination |
| `BillingRepository` | `get_transaction_history()` | `(list, int)` | `list` only (drop count) |
| `UserRepository` | `list_users()` | `(list, int)` | `list` only (drop count) |
| `BillingRepository` | `list_payments()` | `(list, int)` | `list` only (drop count) |

### 2.4 Service Methods That Return `(data, total)` Tuples

| Service | Method | Current Return | New Return |
|---|---|---|---|
| `UserContentService` | `list_user_outputs()` | `(list, int)` | `list` only |
| `UserContentService` | `list_user_uploads()` | `(list, int)` | `list` only |
| `BillingService` | `get_transaction_history()` | `(list, int)` | `list` only |
| `UnifiedJobService` | `list_jobs()` | `PaginatedResponse` (builds internally) | `CursorPage` (new schema) |

---

## 3. New Pagination Schema

### 3.1 Replace `PaginatedResponse`

```python
# src/api/schemas/pagination.py

T = TypeVar("T")


class CursorPage(msgspec.Struct, Generic[T], kw_only=True):
    """Cursor-paginated response — used by all list endpoints.

    No total count. Uses limit+1 fetch to determine has_more.
    No offset parameter — cursor-only pagination.
    """

    items: list[T]
    """Page of results."""

    limit: int
    """Requested page size, echoed back."""

    has_more: bool
    """True when there are additional pages after this one."""

    next_cursor: str | None = None
    """Opaque cursor token for the next page. None when has_more is False."""
```

**Breaking change for frontend.** The response shape changes:

```typescript
// BEFORE
interface PaginatedResponse<T> {
  items: T[];
  total: number;       // ← removed
  limit: number;
  offset: number;      // ← removed
  has_more: boolean;
  next_cursor: string | null;
}

// AFTER
interface CursorPage<T> {
  items: T[];
  limit: number;
  has_more: boolean;
  next_cursor: string | null;
}
```

### 3.2 Keep `PaginatedResponse` as Deprecated Alias (optional, transition period)

If the frontend needs a grace period:

```python
# Deprecated — use CursorPage instead
class PaginatedResponse(msgspec.Struct, Generic[T], kw_only=True):
    items: list[T]
    total: int = 0  # always 0, deprecated
    limit: int
    offset: int = 0  # always 0, deprecated
    has_more: bool
    next_cursor: str | None = None
```

**Recommendation:** Since you're the sole developer on both repos and there's no production DB yet — just cut over cleanly to `CursorPage`. No deprecation period needed.

### 3.3 Keep Cursor Encode/Decode Unchanged

The `encode_cursor()` / `decode_cursor()` functions remain identical — they're already clean.

---

## 4. The `limit + 1` Pattern

Universal pattern for all paginated queries:

```python
async def list_gallery_jobs(
    self,
    user_id: UUID,
    product_id: str,
    *,
    limit: int = 20,
    cursor_ts: datetime | None = None,
    cursor_id: UUID | None = None,
    # ... filters
) -> list[GenerationJob]:
    """Fetch limit+1 rows. Caller checks len > limit for has_more."""

    query = select(GenerationJob).where(
        GenerationJob.user_id == user_id,
        GenerationJob.product_id == product_id,
    )

    if cursor_ts is not None and cursor_id is not None:
        query = query.where(
            sa.tuple_(GenerationJob.created_at, GenerationJob.id) < sa.tuple_(cursor_ts, cursor_id)
        )

    query = query.order_by(
        GenerationJob.created_at.desc(),
        GenerationJob.id.desc(),
    ).limit(limit + 1)  # ← fetch one extra

    result = await self._session.execute(query)
    return list(result.scalars().all())
```

**In the service/controller:**

```python
rows = await repo.list_something(limit=limit, cursor_ts=..., cursor_id=...)

has_more = len(rows) > limit
if has_more:
    rows = rows[:limit]

next_cursor = None
if has_more and rows:
    last = rows[-1]
    next_cursor = encode_cursor(last.created_at, last.id)

return CursorPage(
    items=[to_dto(r) for r in rows],
    limit=limit,
    has_more=has_more,
    next_cursor=next_cursor,
)
```

---

## 5. Drop `offset` Query Parameter

All paginated endpoints currently accept `offset` alongside `cursor`. After this change:

- **Remove** `offset` query parameter from all endpoints.
- **Cursor is the only pagination mechanism.**
- First page: `GET /v1/jobs?limit=20` (no cursor, no offset).
- Next page: `GET /v1/jobs?limit=20&cursor=eyJ...`.

**Exception: Admin endpoints** — admin endpoints could arguably keep offset for "jump to page N" use cases. But since we're removing `total`, page numbers are meaningless anyway. Drop offset everywhere for consistency.

---

## 6. File-by-File Changes

### 6.1 Schema Changes

| File | Change |
|---|---|
| `src/api/schemas/pagination.py` | Replace `PaginatedResponse` with `CursorPage`. Remove `total` and `offset` fields. Keep `encode_cursor` / `decode_cursor`. |
| `src/api/schemas/__init__.py` | Update export: `PaginatedResponse` → `CursorPage` |

### 6.2 Route Changes

| File | Endpoint | Changes |
|---|---|---|
| `src/api/routes/jobs.py` | `GET /v1/jobs` | Remove `offset` param. Return `CursorPage`. |
| `src/api/routes/storage.py` | `GET /v1/storage/outputs` | Remove `offset` param. Remove `total` from response. Use `limit+1`. |
| `src/api/routes/storage.py` | `GET /v1/storage/uploads` | Same. |
| `src/api/routes/storage.py` | `GET /v1/storage/jobs/{id}/outputs` | Already no pagination — no change needed. |
| `src/api/routes/billing.py` | `GET /v1/billing/transactions` | Remove `offset` param. Return `CursorPage`. |
| `src/api/routes/admin.py` | `GET /v1/admin/users` | Remove `offset` param. Add `cursor` param. Return `CursorPage`. |
| `src/api/routes/admin.py` | `GET /v1/admin/payments` | Remove `offset` param. Add `cursor` param. Return `CursorPage`. |
| `src/api/routes/admin.py` | `GET /v1/admin/accounts/{id}/transactions` | Remove `offset` param. Return `CursorPage`. |

### 6.3 Service Changes

| File | Method | Changes |
|---|---|---|
| `src/api/services/unified_jobs.py` | `list_jobs()` | Remove `offset` param. Remove COUNT query. Fetch `limit+1`. Return `CursorPage` directly. |
| `src/api/services/user_content.py` | `list_user_outputs()` | Remove COUNT call (`count_user_outputs`). Return `list` only (not tuple). |
| `src/api/services/user_content.py` | `list_user_uploads()` | Remove COUNT call (`count_user_images`). Return `list` only (not tuple). |
| `src/api/services/billing.py` | `get_transaction_history()` | Remove COUNT passthrough. Return `list` only (not tuple). |

### 6.4 Repository Changes

| File | Method | Changes |
|---|---|---|
| `src/db/repositories/storage.py` | `list_user_outputs()` | Remove `offset` param. Fetch `limit+1`. |
| `src/db/repositories/storage.py` | `list_user_images()` | Remove `offset` param. Fetch `limit+1`. |
| `src/db/repositories/storage.py` | `count_user_outputs()` | **Mark for deletion** — no longer called by pagination. Keep only if `get_user_storage_stats()` still needs it (it doesn't — stats uses its own inline COUNT). |
| `src/db/repositories/storage.py` | `count_user_images()` | **Mark for deletion** — same reasoning. |
| `src/db/repositories/billing.py` | `get_transaction_history()` | Remove COUNT query. Remove `offset` param. Fetch `limit+1`. Return `list` only. |
| `src/db/repositories/user.py` | `list_users()` | Remove COUNT query. Remove `offset` param. Add cursor params. Fetch `limit+1`. Return `list` only. |
| `src/db/repositories/billing.py` | `list_payments()` | Remove COUNT query. Remove `offset` param. Add cursor params. Fetch `limit+1`. Return `list` only. |

### 6.5 Test Changes

| File | Changes |
|---|---|
| `tests/unit/test_pagination.py` | Rewrite for `CursorPage` schema. Remove `total`/`offset` assertions. Test `limit+1` → `has_more` derivation. |
| `tests/unit/test_jobs.py` | Update `list_jobs` tests: remove `total` field, mock session to not return count. |
| `tests/integration/test_repository_*` | Remove assertions on `total` from `(data, total)` returns. |
| `tests/integration/test_user_repository.py` | Update `list_users` assertions (no total in return). |
| `docs/BACKEND_API_REFERENCE.md` | Update pagination section, all endpoint docs. |

---

## 7. Migration Strategy

Since there is no production database and the frontend is under sole developer control, this is a clean cutover:

1. **Backend:** Replace `PaginatedResponse` → `CursorPage` in one PR.
2. **Frontend:** Update the `PaginatedResponse<T>` TypeScript interface to `CursorPage<T>` — remove `total` and `offset` fields.
3. **No backward compatibility layer needed.**

### Ordering (within the gallery implementation prompt)

This pagination refactor should be **Phase 0** of the gallery implementation — it's a prerequisite since the gallery uses `CursorPage` from the start, and doing it first means the gallery doesn't introduce a second pagination pattern.

---

## 8. Updated API Contract

### Before

```
GET /v1/jobs?limit=20&offset=40
GET /v1/jobs?limit=20&cursor=eyJ...

Response: {
  "items": [...],
  "total": 142,
  "limit": 20,
  "offset": 40,
  "has_more": true,
  "next_cursor": "eyJ..."
}
```

### After

```
GET /v1/jobs?limit=20
GET /v1/jobs?limit=20&cursor=eyJ...

Response: {
  "items": [...],
  "limit": 20,
  "has_more": true,
  "next_cursor": "eyJ..."
}
```

Query parameters on all paginated endpoints:

```
limit?    int    Page size (1–N, endpoint-specific default)
cursor?   str    Opaque cursor from previous response's next_cursor
```

No `offset`. No `total`. Cursor-only pagination everywhere.

---

## 9. Methods to Delete

These repository methods exist solely to serve pagination COUNT queries and have no other callers after the refactor:

| Repository | Method | Safe to Delete? |
|---|---|---|
| `StorageRepository.count_user_outputs()` | Yes — `get_user_storage_stats()` uses its own inline `COUNT` + `SUM` |
| `StorageRepository.count_user_images()` | Yes — same reasoning |

The `UserRepository` count methods (`get_user_output_count`, `get_user_upload_count`, `get_user_job_count`, `count_job_outputs`) are used by admin/analytics features, **not** pagination — keep them.

---

## 10. Keyset Pagination Correctness

The existing keyset WHERE clause uses `OR` with `AND`:

```python
where(
    or_(
        Model.created_at < cursor_ts,
        and_(
            Model.created_at == cursor_ts,
            Model.id < cursor_id,
        ),
    )
)
```

This is correct but verbose. The `tuple_()` form is cleaner and generates the same SQL:

```python
from sqlalchemy import tuple_

where(tuple_(Model.created_at, Model.id) < tuple_(cursor_ts, cursor_id))
```

PostgreSQL optimizes `(a, b) < (x, y)` into the same plan as the `OR`/`AND` form.

**Recommendation:** Standardize on `tuple_()` across all repositories for consistency and readability. The migration touches all these methods anyway.

---

## 11. Summary of Breaking Changes

| Change | Frontend Impact |
|---|---|
| `total` field removed from all paginated responses | Remove from TypeScript interface, stop displaying "X total results" |
| `offset` query parameter removed from all endpoints | Stop sending `offset=N`, use only `cursor=` |
| Response type renamed `PaginatedResponse` → `CursorPage` | Update TypeScript type name (or just update the interface shape) |
| Admin endpoints now require `cursor` for paging (no offset) | Update admin table pagination to cursor-based |
