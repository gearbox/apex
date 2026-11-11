# Frontend Contract — Media Project Studio (S1-P1: Studio Foundation)

> **Audience:** `gearbox/apex-frontend` (artists, studio projects, track editor, version history, Suno export).
> **Backend source:** branch `feat/studio-foundation` (migration `055_studio_foundation`).
> **Authority:** `gen:api` (OpenAPI `schema.json`) is authoritative for **types**. This document is authoritative for **semantics**: limits, normalization, error codes, idempotency, the "current pair" rules and what the Suno export does and does not contain.
> **Scope of this phase:** artists, a studio profile on a library project, tracks, versioned Plan / Lyrics documents, a deterministic Suno export. **No LLM planner, no audio, no Cast** — those arrive in later phases and will extend, not break, this contract.

---

## 0. Summary

- The user-facing container is the existing **library project**. Enabling studio on a project (`PUT /v1/studio/projects/{id}`) attaches a *studio profile* (release kind, artist, concept) to it; the project keeps its name and assets.
- A project has up to 64 **tracks**, ordered. A track has two independently versioned documents: a **Plan** (structure: sections, tags, tempo) and **Lyrics** (lines per plan section).
- Every save of a document is a new **immutable version** in a tree (`parent_version_id`). Versions are never edited. The track points at its **current** plan and lyrics.
- **Version create is idempotent by content.** Re-sending the same content under the same parent returns the existing version (`200`) instead of creating a duplicate (`201`). You may retry freely.
- The current **(plan, lyrics) pair is always compatible**. The server refuses (`409`) any action that would make an incompatible pair current, and a refused action writes nothing.
- `GET …/exports/suno` renders Suno's **Title / Style / Exclude / Lyrics** fields. It always renders; it never truncates; limits are reported, not enforced.
- Everything sits behind a feature flag: when studio is off for the product (or globally) **every** `/v1/studio/*` route answers `404`, even before authentication.

---

## 1. Feature gating

```
STUDIO_ENABLED=true  AND  product has the "studio" feature   →  routes exist
otherwise                                                    →  404 {"error": "not_found", …}
```

- Both `vex` and `synthara` list the `studio` feature; `STUDIO_ENABLED` (default **off**) is the production switch.
- The check runs **before** authentication: an unauthenticated call to a disabled studio gets `404`, not `401`. Treat a `404` on any studio route as "the feature is off" and hide the studio UI.
- Library routes (`/v1/library/*`) are never affected by the switch.

All studio mutations are subject to the same **legal-acceptance enforcement** as other mutating routes (`428 legal_acceptance_required` until the user re-accepts). Reads are never blocked.

---

## 2. Types

```ts
type UUID = string;
type ISODateTime = string;

type ReleaseKind = "single" | "ep" | "album";
type DocumentKind = "plan" | "lyrics";
type DocumentOrigin = "user" | "llm" | "import";   // only "user" is written in this phase
type SectionKind =
  | "intro" | "verse" | "pre_chorus" | "chorus" | "post_chorus" | "hook" | "bridge"
  | "build" | "drop" | "breakdown" | "instrumental" | "interlude" | "outro";

// ----- artists -------------------------------------------------------------------------
interface Artist {
  id: UUID;
  name: string;                      // 1–100 after normalization, unique per owner (case-insensitive via lower(), see §9)
  persona: string | null;            // ≤ 4000 — planner context only, never exported
  voice_notes: string | null;        // ≤ 2000 — planner context only, never exported
  default_language: string | null;   // BCP-47-like, e.g. "en", "pt-BR"
  aesthetic_references: string[];    // ≤ 20 × ≤ 200 — planner context only, NEVER exported
  style_tags: string[];              // ≤ 40 × ≤ 60 — EXPORTED (vocal type, delivery …)
  global_avoid: string[];            // ≤ 40 × ≤ 60 — EXPORTED into Suno's Exclude
  created_at: ISODateTime;
  updated_at: ISODateTime;
}
interface ArtistCreate {             // unknown fields → 400
  name: string;
  persona?: string | null; voice_notes?: string | null; default_language?: string | null;
  aesthetic_references?: string[]; style_tags?: string[]; global_avoid?: string[];
}
type ArtistPatch = Partial<ArtistCreate>;   // omitted = keep; null clears persona/voice_notes/default_language

// ----- projects & tracks ---------------------------------------------------------------
interface ProjectProfile {
  library_project_id: UUID;          // the library project's id IS the studio project id
  name: string;                      // the library project's name
  release_kind: ReleaseKind;
  artist_id: UUID | null;
  concept: string | null;            // ≤ 8000, album-level context
  track_count: number;
  created_at: ISODateTime; updated_at: ISODateTime;
}
interface ProjectProfilePut { release_kind: ReleaseKind; artist_id?: UUID | null; concept?: string | null }  // full replace

interface Track {
  id: UUID;
  library_project_id: UUID;
  position: number;                  // 0-based; gaps are possible after a delete until you reorder
  title: string;                     // 1–200
  current_plan_version_id: UUID | null;
  current_lyrics_version_id: UUID | null;   // never set without a plan
  created_at: ISODateTime; updated_at: ISODateTime;
}
interface TrackDetail extends Track {
  current_plan: PlanVersion | null;
  current_lyrics: LyricsVersion | null;
}
```

### 2.1 Documents

```ts
interface StyleSpec { tags?: string[]; avoid?: string[] }          // each ≤ 40 × ≤ 60 chars

interface PlanSection {
  id: UUID;                          // chosen by the client, STABLE across plan versions
  kind: SectionKind;
  label?: string | null;             // 1–60; shown in the Suno header instead of the kind name
  duration_seconds?: number | null;  // 1–600, planned only — NOT exported
  style?: StyleSpec;                 // section-local tags (avoid is NOT exportable)
  delivery?: string[];               // ≤ 12 × (1–40)
  instrumental?: boolean;            // default false; an instrumental section carries no lyrics
  notes?: string | null;             // ≤ 1000 — NEVER exported
}
interface PlanDocument {             // schema_version 1; unknown fields → 400
  title: string;                     // 1–200
  idea?: string | null;              // ≤ 4000 — NEVER exported
  tempo_bpm?: number | null;         // 40–240
  global_style?: StyleSpec;
  references?: string[];             // ≤ 20 × (1–200) — NEVER exported (real-artist names)
  sections: PlanSection[];           // 1–64
}

interface LyricLine { text: string /* 1–300 */; delivery?: string[] /* ≤ 6 × (1–40) */ }
interface LyricsSection { section_id: UUID; lines: LyricLine[] /* 1–64 */ }
interface LyricsDocument {           // schema_version 1
  language: string;                  // /^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$/, ≤ 35
  line_syllables?: { min: number; max: number } | null;   // 1–32; stored, enforced by a later phase
  sections?: LyricsSection[];        // ≤ 64; at most one entry per plan section
}
```

**Size limits (checked after normalization, reported as `422 document_invalid` issues — see §4).** The per-field limits above alone would allow megabytes per version, so two totals apply on top of them:

| Limit | Value | Issue code (path) |
|---|---|---|
| Lyrics text — the sum of `text` over **all** lines of all sections, in Unicode code points | **20,000** | `lyrics_too_long` (`sections`) |
| Any document (plan or lyrics) as compact UTF-8 JSON — what is stored and hashed | **131,072** bytes (128 KiB) | `document_too_large` (`""`, the document root) |

20,000 characters is about four times Suno's own Lyrics box. Cyrillic text costs about 2 bytes per character (20,000 characters in 400 lines ≈ 52 KB), so the character limit is the one a normal lyricist meets first; `document_too_large` is for pathological shapes (thousands of 1–4-character lines, or very large plans). Both limits are measured on the *normalized* form, so padding whitespace does not count. A document exactly at a limit is accepted.

### 2.2 Versions

```ts
interface VersionSummary {           // list items — no content
  id: UUID; kind: DocumentKind;
  version_number: number;            // per (track, kind); starts at 1; NEVER reused, even after a delete
  parent_version_id: UUID | null;
  based_on_plan_version_id: UUID | null;   // set for lyrics only
  origin: DocumentOrigin; schema_version: number; content_hash: string;
  created_at: ISODateTime; is_current: boolean;
}
type PlanVersion   = { kind: "plan";   id: UUID; track_id: UUID; version_number: number; parent_version_id: UUID | null;
                       origin: DocumentOrigin; schema_version: number; content_hash: string; created_at: ISODateTime;
                       is_current: boolean; content: PlanDocument };
type LyricsVersion = { kind: "lyrics"; id: UUID; track_id: UUID; version_number: number; parent_version_id: UUID | null;
                       based_on_plan_version_id: UUID; origin: DocumentOrigin; schema_version: number;
                       content_hash: string; created_at: ISODateTime; is_current: boolean; content: LyricsDocument };

// POST /v1/studio/tracks/{id}/versions — tagged union on "kind"
type VersionCreate =
  | { kind: "plan";   content: PlanDocument;   parent_version_id?: UUID | null; make_current?: boolean /* true */;
      incompatible_lyrics?: "reject" | "detach" /* "reject" */ }
  | { kind: "lyrics"; content: LyricsDocument; based_on_plan_version_id: UUID; parent_version_id?: UUID | null;
      make_current?: boolean /* true */ };
interface VersionCreated { version: PlanVersion | LyricsVersion; lyrics_detached: boolean }
```

---

## 3. Endpoints

All paths are under `/v1/studio`; all need a bearer token and answer `404` when studio is off. Lists use cursor pagination (`?limit=&cursor=` → `{items, limit, has_more, next_cursor}`); a bad cursor is `400 invalid_cursor`.

| Method | Path | Success | Notes |
|---|---|---|---|
| GET | `/artists` | 200 page | `limit` 1–50 (30) |
| POST | `/artists` | 201 `Artist` | max **50** artists per user |
| GET / PATCH | `/artists/{id}` | 200 `Artist` | |
| DELETE | `/artists/{id}` | 204 | `409 artist_in_use` while a project uses it |
| GET | `/projects` | 200 page of `ProjectProfile` | newest first |
| GET | `/projects/{id}` | 200 `ProjectProfile` | `404` if studio is not enabled on that project |
| PUT | `/projects/{id}` | **201** created / **200** replaced | `{id}` must be one of the caller's library projects |
| DELETE | `/projects/{id}` | 204 | disables studio only; `409 project_has_tracks` if tracks exist |
| GET | `/projects/{id}/tracks` | 200 `Track[]` | ordered, not paginated (≤ 64) |
| POST | `/projects/{id}/tracks` | 201 `Track` | appended; max **64** per project |
| PUT | `/projects/{id}/tracks/order` | 200 `Track[]` | body `{track_ids}` — the exact set, in the new order; positions become `0..n-1` |
| GET / PATCH / DELETE | `/tracks/{id}` | 200 `TrackDetail` / 200 `Track` / 204 | delete removes the track's versions too |
| GET | `/tracks/{id}/versions` | 200 page of `VersionSummary` | `?kind=plan\|lyrics`, `limit` 1–100 (30), newest first |
| POST | `/tracks/{id}/versions` | **201** created / **200** replay | see §4 |
| GET | `/versions/{id}` | 200 `PlanVersion \| LyricsVersion` | |
| DELETE | `/versions/{id}` | 204 | see §6 |
| PUT | `/tracks/{id}/current` | 200 `Track` | body `{plan_version_id?, lyrics_version_id?}` — see §5 |
| GET | `/tracks/{id}/exports/suno` | 200 `SunoExport` | `?plan_version_id=&lyrics_version_id=` — see §8 |

Also: deleting a **library project** that still has studio tracks is refused — `409 project_in_use` with
`detail: {"blockers": [{"code": "studio_tracks", "count": n}]}`. Delete (or move) the tracks first. A studio project without tracks deletes normally (its profile goes with it).

CORS: studio uses `PUT`; it is included in the API's allowed methods.

---

## 4. Creating versions (idempotent)

1. **Normalization.** Before anything else the server normalizes every tag, label and line: Unicode **NFC**, trim, collapse internal whitespace to one space. The *normalized* form is stored and hashed. `"Café  Noir"` (NFD, double space) and `" Café Noir"` (NFC) are the same content.
2. **Validation returns every problem at once** — `422 document_invalid`, `detail.issues = [{path, code}]`. Paths look like `sections[2].style.tags[1]`; they never contain your text. Codes:

   | Code | Meaning |
   |---|---|
   | `empty_after_normalization` | blank once trimmed (title, label, tag, line text, …) |
   | `duplicate_section_id` | two plan sections share an `id` |
   | `duplicate_tag` | same tag twice in one list (case-insensitive, also non-ASCII) |
   | `tag_conflict` | a tag is in both `tags` and `avoid` of the same style |
   | `reserved_characters` | `[` or `]` in a tag, delivery tag, label or lyric line (they would break Suno meta-tags) |
   | `total_duration_exceeded` | planned section durations sum to more than 1800 s |
   | `unknown_section` | lyrics refer to a section the plan does not have |
   | `duplicate_section` | lyrics list the same section twice |
   | `instrumental_section_has_lines` | lyrics on an instrumental section |
   | `syllable_range_inverted` | `line_syllables.min > max` |
   | `lyrics_too_long` | the lyrics' line text totals more than 20,000 characters (path `sections`) |
   | `document_too_large` | the normalized document is larger than 131,072 bytes as compact JSON (path `""`, the root) |
   | `requires_make_current` | `incompatible_lyrics: "detach"` was sent with `make_current: false` (path `incompatible_lyrics`); checked first, before anything else is read |

   The size codes are collected together with every other issue; they never short-circuit validation.

   Wire-level violations (lengths, ranges, unknown fields, bad enum, malformed union) are `400`.
3. **Idempotency.** A version is identified by `(track, kind, parent_version_id, normalized content [, based_on_plan_version_id])`.
   - First time → `201`. Same again → **`200` with the existing version**, no new row, no number consumed.
   - A replay **still applies `make_current`**: the call means "this version exists and, if asked, is current". Re-sending an old version's content with `make_current: true` is a valid way to switch back to it.
   - No `Idempotency-Key` header is needed (or read).
4. **No-op saves are refused.** Content equal to its parent's → `409 version_unchanged`.
5. **Parents and bases.** `parent_version_id` must be a version of the *same track and kind* (`404 parent_version_not_found`). A lyrics version's `based_on_plan_version_id` must be a *plan* version of the same track (`404 plan_version_not_found`). Lyrics are validated against the plan they are based on.
6. **Caps.** 500 versions per (track, kind) → `409 version_limit_reached` (replays are exempt; deleting a version frees room).

Retry rule: any network failure can be retried verbatim — the worst case is a `200` replay.

---

## 5. The current (plan, lyrics) pair

A track has at most one current plan and one current lyrics. Invariants the server enforces for you:

- Lyrics are never current without a plan (`409 plan_required`).
- The current pair is always **compatible**: every lyrics section exists in the plan and none sits on an instrumental section.
- A `409` here is **atomic** — nothing is written (no row, no number, no pointer).

| Action | Rule |
|---|---|
| Create **lyrics**, `make_current: true` | needs a current plan (`409 plan_required`); checked against the **current** plan, not only its base (`409 lyrics_incompatible_with_plan`, `detail.issues`). With `make_current: false` it is stored without checks against the current plan. |
| Create **plan**, `make_current: true` | if the current lyrics no longer fit: `incompatible_lyrics: "reject"` (default) → `409 lyrics_incompatible_with_plan`; `"detach"` → the plan becomes current, the **lyrics pointer is cleared** (the lyrics version itself is kept) and the response has `lyrics_detached: true`. With compatible lyrics `detach` changes nothing. `"detach"` **requires `make_current: true`**: with `make_current: false` there is no change to the pair, so the request is refused with `422 document_invalid` (`{path: "incompatible_lyrics", code: "requires_make_current"}`) and nothing is written. |
| `PUT …/current` | each given id must be a version of this track **of the matching kind** (`404` otherwise). Omitted fields keep their pointer; `lyrics_version_id: null` detaches the lyrics. The resulting pair must be compatible. To move to a plan with different sections send `{plan_version_id: X, lyrics_version_id: null}`. |

There is deliberately **no silent auto-detach**: dropping the lyrics is always the caller's explicit choice.

---

## 6. Deleting versions

`DELETE /versions/{id}` succeeds only if the version is **not current**, has **no children** and is **not the base of a lyrics version**:

| Refusal | Code | `detail` |
|---|---|---|
| it is the track's current plan or lyrics | `409 version_is_current` | |
| other versions list it as `parent_version_id` | `409 version_has_children` | `{count}` |
| lyrics versions are based on this plan | `409 version_is_lyrics_base` | `{count}` |

Version numbers are never reused: after deleting #3, the next version is #4.

---

## 7. Other errors

Every error is the unified envelope `{error, message, status_code, detail}`. `message` is static English text — use `error` for logic.

| Status | `error` | When |
|---|---|---|
| 400 | `invalid_cursor` | malformed `cursor` |
| 404 | `not_found` | missing, someone else's, or another product's resource (never distinguishable) |
| 404 | `artist_not_found` | `artist_id` is not one of the caller's artists |
| 409 | `artist_name_conflict` | name already used (case-insensitive) |
| 409 | `artist_limit_reached` | 50 artists |
| 409 | `artist_in_use` | `detail: {project_count}` |
| 409 | `project_has_tracks` | disabling studio on a project with tracks |
| 409 | `track_limit_reached` | 64 tracks |
| 422 | `track_order_mismatch` | `track_ids` is not exactly the project's track set (missing, extra or duplicate id) |
| 422 | `document_invalid` | §4 |
| 428 | `legal_acceptance_required` | stale legal acceptance on a mutation |

---

## 8. Suno export

`GET /v1/studio/tracks/{id}/exports/suno[?plan_version_id=&lyrics_version_id=]` — a pure render: it reads, never writes, and the same input yields byte-identical output.

- **The plan and the lyrics are chosen as a pair.** The track's current lyrics were written for its current plan, so they are used *only* when you name no plan:

  | `plan_version_id` | `lyrics_version_id` | Rendered |
  |---|---|---|
  | omitted | omitted | the track's current plan **and** current lyrics (no current lyrics → the section skeleton) |
  | given | omitted | that plan **alone** (the section skeleton, `lyrics_version_id: null`); the current lyrics are not borrowed |
  | omitted | given | the track's current plan with those lyrics |
  | given | given | exactly that pair — the caller's choice, even across plans |

  `409 plan_required` if no plan is determined (no `plan_version_id` and no current plan). Explicit ids must be versions of this track and the right kind (`404`). Whatever lyrics are rendered are checked against the plan they are rendered with (`409 lyrics_incompatible_with_plan`), so an explicit pair that does not fit is refused. The response's `lyrics_version_id` is always exactly the lyrics that were rendered (`null` when none).

```ts
interface SunoExport {
  target: "suno";
  renderer_version: string;          // e.g. "suno-v6.2026-11" — changes when rendering rules change
  plan_version_id: UUID;
  lyrics_version_id: UUID | null;
  title: string;                     // = plan title
  style: string;                     // paste into Suno "Style of Music"
  exclude: string;                   // paste into "Exclude styles"
  lyrics: string;                    // paste into "Lyrics"
  fields: { title | style | exclude | lyrics: { length: number; limit: number | null; within_limit: boolean } };
  warnings: { code: string; section_id: UUID | null }[];
}
```

**Rendering rules**

- `style` = plan `global_style.tags`, then the project artist's `style_tags`, de-duplicated case-insensitively (first occurrence wins), then `"<tempo> BPM"`. `exclude` = plan `global_style.avoid` then artist `global_avoid`, same de-duplication.
- `lyrics` = one block per plan section, in plan order, separated by a blank line. Header `[Label]` or `[Label: descriptors]` where descriptors are the section's `delivery` then its `style.tags`. Label = the section's `label`, else the kind's name (`Pre-Chorus`, `Post-Chorus`, otherwise title-case); a defaulted label is numbered (`Verse 1`, `Verse 2`) when the kind occurs more than once — except `chorus`, `pre_chorus`, `post_chorus` and `hook`, which stay unnumbered. A line with `delivery` is preceded by its own `[d1, d2]` line. **An instrumental section is never rendered without a cue:** a section with `instrumental: true` whose kind is not `instrumental` gets the descriptor `instrumental` **first** in its header (`[Drop: instrumental, heavy bass]`), de-duplicated case-insensitively against its own delivery and style tags, so a long tag list cannot push it out of sight; a section of kind `instrumental` is already headed `[Instrumental]` and gets no extra cue. **No parentheses are ever generated** (Suno sings them as backing vocals); `(…)` appears only if you typed it.
- **Never exported:** `references`, `idea`, section `notes`, `duration_seconds`, section-local `avoid`, the artist's `persona`, `voice_notes` and `aesthetic_references`.

**Limits are information, not errors.** Suno publishes no limits; the values below were measured on the v6 web UI and may change. The export **never truncates and never fails on length** — the full text is returned and `within_limit: false` tells you to cut:

| Field | `limit` |
|---|---|
| `title` | `null` (unknown) |
| `style` | 1000 |
| `exclude` | 1000 |
| `lyrics` | 5000 |

`length` counts Unicode code points.

**Warnings** (stable codes; show them, don't treat them as failures):

| Code | `section_id` | Meaning |
|---|---|---|
| `local_avoid_not_exportable` | yes | the section has `style.avoid`, which Suno cannot express per section |
| `section_duration_not_exportable` | yes | the section has a planned duration, which Suno cannot take |
| `lyrics_above_recommended_length` | no | lyrics exceed 3000 characters; Suno tends to rush longer lyrics |
| `style_exclude_overlap` | no | a tag ended up in both Style and Exclude after merging the artist's lists |

---

## 9. Behavioural notes for the UI

- **Section ids are yours.** Generate a UUID per plan section once and keep it across plan versions — lyrics reference sections by id, so renaming or reordering sections never orphans lyrics.
- **Open a track with one call:** `GET /tracks/{id}` returns the current plan and lyrics.
- **Branching history:** every version has `parent_version_id`; a "version tree" view is a straightforward render of `GET /tracks/{id}/versions?kind=…`. Deleting a leaf is allowed; deleting a node with children is not.
- **Optimistic UI is safe.** Because create is idempotent, a double submit yields one version.
- **Positions** can have gaps after a track delete; use `position` for ordering only and call the order endpoint to compact.
- The studio profile's `artist_id` change (via `PUT`) affects the *next* export immediately — exports always read the artist's current style.
- **Artist names are compared with `lower()`, not `casefold()`** (the same rule as library project names; PostgreSQL has no `casefold`). Plain case differences collide (`Nova` / `NOVA`, and non-ASCII letters such as `Рок` / `рок` under the default UTF-8 database locale), but a few characters that only casefold equates stay distinct: `Straße` and `STRASSE` are two valid names. Tags, labels and the other free-text lists use `casefold` (`duplicate_tag` catches `ß` / `SS`). This is a known, accepted deviation; do not rely on name uniqueness being casefold-strict.
