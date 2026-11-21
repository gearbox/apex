# Frontend Contract — Media Project Studio (S1-P1 / S1-P1b / S1-P2a / S1-P2b: Studio Foundation, Suno import, Planner and lyrics runs, Artist memories)

> **Audience:** `gearbox/apex-frontend` (artists, studio projects, track editor, version history, Suno export and import).
> **Backend source:** branches `feat/studio-foundation` (migration `055_studio_foundation`), `feat/studio-suno-import` (migration `056_studio_plan_schema_v2`, plan **schema v2**) `feat/studio-planner-runs` (migration `057_studio_runs`), `feat/studio-lyrics-runs` (migration `058_studio_run_inputs`) and `feat/studio-artist-memories` (migration `059_studio_artist_memories`).
> **Authority:** `gen:api` (OpenAPI `schema.json`) is authoritative for **types**. This document is authoritative for **semantics**: limits, normalization, error codes, idempotency, the "current pair" rules and what the Suno export does and does not contain.
> **Scope of this phase:** artists, a studio profile on a library project, tracks, versioned Plan / Lyrics documents, a deterministic Suno export and its inverse, a deterministic Suno-text import (§10). and — behind a separate switch — LLM **planner runs** that draft a plan, write lyrics or revise them (§11), grounded in an artist's **memories** (§12). **No artist chat, no audio, no Cast** — those arrive in later phases and will extend, not break, this contract.

---

## 0. Summary

- The user-facing container is the existing **library project**. Enabling studio on a project (`PUT /v1/studio/projects/{id}`) attaches a *studio profile* (release kind, artist, concept) to it; the project keeps its name and assets.
- A project has up to 64 **tracks**, ordered. A track has two independently versioned documents: a **Plan** (structure: sections, tags, tempo) and **Lyrics** (lines per plan section).
- Every save of a document is a new **immutable version** in a tree (`parent_version_id`). Versions are never edited. The track points at its **current** plan and lyrics.
- **Version create is idempotent by content.** Re-sending the same content under the same parent returns the existing version (`200`) instead of creating a duplicate (`201`). You may retry freely.
- The current **(plan, lyrics) pair is always compatible**. The server refuses (`409`) any action that would make an incompatible pair current, and a refused action writes nothing.
- `GET …/exports/suno` renders Suno's **Title / Style / Exclude / Lyrics** fields. It always renders; it never truncates; limits are reported, not enforced.
- `POST …/imports/suno` is the inverse: paste Suno's Style / Exclude / Lyrics boxes back and the server turns them into the track's next plan and lyrics versions (`origin: "import"`). Deterministic, no LLM, free; `dry_run` previews without writing (§10).
- An artist can hold up to 200 **memories** (§12): episodes of its fictional life that planner runs use as *inspiration*. You pin the ones you want on a run and let the server fill the rest by salience, within a fixed budget.
- A plan has a track-level **`instrumental`** flag (independent of its sections and of the lyrics) and may have **no sections** at all (plan **schema v2**, §2.1).
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
type DocumentOrigin = "user" | "llm" | "import";   // "user" (typed) and "import" (§10) are written; "llm" arrives later
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
  style_tags: string[];              // ≤ 40 × ≤ 60 — EXPORTED (vocal type, delivery …); an item that reads as a tempo ("120 BPM") is refused, see `reserved_tempo_tag` (§4)
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

// ----- artist memories (§12) -----------------------------------------------------------
type MemoryStatus = "active" | "archived";
interface Memory {
  id: UUID;
  artist_id: UUID;
  title: string;                     // 1–120 after normalization, one line
  body: string;                      // 1–2000 after normalization, line breaks kept
  occurred_on: string | null;        // "YYYY-MM-DD": a fictional date, used for ordering and display only
  themes: string[];                  // ≤ 12 × ≤ 60
  imagery: string[];                 // ≤ 20 × ≤ 120 — concrete sensory details
  salience: number;                  // 1–5, default 3
  status: MemoryStatus;
  origin: "user" | "chat";           // "user" today; "chat" arrives with the artist chat
  created_at: ISODateTime; updated_at: ISODateTime;
}
interface MemoryCreate {             // unknown fields (including `origin` and `status`) → 400
  title: string; body: string; occurred_on?: string | null;
  themes?: string[]; imagery?: string[]; salience?: number;
}
type MemoryPatch = Partial<MemoryCreate> & { status?: MemoryStatus };   // omitted = keep; null clears occurred_on

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
interface PlanDocument {             // schema_version 2; unknown fields → 400
  title: string;                     // 1–200
  idea?: string | null;              // ≤ 4000 — NEVER exported
  tempo_bpm?: number | null;         // 40–240; the ONLY place for a tempo: a global tag such as "120 BPM" is refused (`reserved_tempo_tag`, §4)
  instrumental?: boolean;            // default false; the TRACK is wordless — see below
  global_style?: StyleSpec;
  references?: string[];             // ≤ 20 × (1–200) — NEVER exported (real-artist names)
  sections?: PlanSection[];          // 0–64 (default []); a plan may be only a title and a style
}

interface LyricLine { text: string /* 1–300 */; delivery?: string[] /* ≤ 6 × (1–40) */ }
interface LyricsSection { section_id: UUID; lines: LyricLine[] /* 1–64 */ }
interface LyricsDocument {           // schema_version 1
  language: string;                  // /^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$/, ≤ 35
  line_syllables?: { min: number; max: number } | null;   // 1–32; stored, enforced by a later phase
  sections?: LyricsSection[];        // ≤ 64; at most one entry per plan section
}
```

**Plan schema v2 (migration 056).** `instrumental` is a property of the *track*: it is versioned with the plan and is independent of every section's own `instrumental` flag and of the lyrics. It is **never inferred** — an empty Lyrics box or a plan of headers only is not an instrumental. A plan marked `instrumental` cannot be paired with lyrics that have any line (§5); lyrics with no lines are fine. `sections` used to need at least one entry and is now optional. Versions stored before this release were rewritten by the migration (`instrumental: false`) and carry `schema_version: 2`; clients that do not send `instrumental` get `false`.

**Size limits (checked after normalization, reported as `422 document_invalid` issues — see §4).** The per-field limits above alone would allow megabytes per version, so two totals apply on top of them:

| Limit | Value | Issue code (path) |
|---|---|---|
| Lyrics text — the sum of `text` over **all** lines of all sections, in Unicode code points | **20,000** | `lyrics_too_long` (`sections`) |
| Any document (plan or lyrics) as compact UTF-8 JSON — what is stored and hashed | **131,072** bytes (128 KiB) | `document_too_large` (`""`, the document root) |

20,000 characters is about four times Suno's own Lyrics box. Cyrillic text costs about 2 bytes per character (20,000 characters in 400 lines ≈ 52 KB), so the character limit is the one a normal lyricist meets first; `document_too_large` is for pathological shapes (thousands of 1–4-character lines, or very large plans). Both limits are measured on the *normalized* form, so padding whitespace does not count. A document exactly at a limit is accepted.

**Every per-field limit above is counted in Unicode code points *after* NFC normalization.** The wire check sees the text as sent; NFC can then make some characters longer (U+0958 becomes 2 code points, U+FB2C becomes 3), so a value that was within its limit on the wire can be over it once normalized. Such a value is refused with `too_long_after_normalization` at the field's path, together with every other issue — it is never stored and never truncated. The same re-check applies to artist, track and project-profile fields (`name`, `persona`, `voice_notes`, `aesthetic_references[i]`, `style_tags[i]`, `global_avoid[i]`, track `title`, `concept`): those requests answer `422 document_invalid` with the same `detail.issues` shape. Text that is already NFC (nearly all Latin, Cyrillic, CJK) is unaffected.

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
| GET | `/artists/{id}/memories` | 200 page of `Memory` | `?status=active\|archived` (default `active`), `limit` 1–50 (30), newest first — see §12 |
| POST | `/artists/{id}/memories` | 201 `Memory` | max **200 active** per artist; `409 memory_limit_reached` — see §12 |
| GET / PATCH | `/memories/{id}` | 200 `Memory` | PATCH also archives / restores (`status`) |
| DELETE | `/memories/{id}` | 204 | always allowed, even while a run uses it — see §12 |
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
| POST | `/tracks/{id}/imports/suno` | **201** written / **200** not | paste Suno's boxes back, `dry_run` to preview — see §10 |

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
   | `reserved_descriptor` | a section's `delivery` or `style.tags` item is the word `instrumental` (any case; path `sections[i].delivery[j]` / `sections[i].style.tags[j]`). Use the section's `instrumental` flag: the export writes the cue itself, and a tag with that word could not be told apart from it when a header is read back (§10) |
   | `reserved_tempo_tag` | a plan `global_style.tags` item (path `global_style.tags[i]`) or an artist `style_tags` item (path `style_tags[i]`) reads as a **tempo**: `120 BPM`, `120bpm`, `120 bpm` (any case, optional space) with a value of 40–240. The tempo belongs in `tempo_bpm`, which the export writes as the last Style item and the importer (§10) reads back from the first such item — a tag with the same spelling would turn into the tempo on the way back. Only these two lists are checked (not section tags, delivery or any avoid list); `300 BPM` or `39 BPM` is an ordinary tag |
   | `reserved_delivery_tag` | a lyric line's delivery, written as the bracket line `[d1, d2]`, would read as a **section header** — its text before the first `:` is a section name such as `Verse`, `Chorus 2`, `Refrain`, `Fade Out` (path `sections[i].lines[j].delivery`). Pick another word |
   | `instrumental_plan_has_lines` | the plan is `instrumental` and the lyrics have at least one line (path `sections`; one issue, besides the per-section ones) |
   | `total_duration_exceeded` | planned section durations sum to more than 1800 s |
   | `unknown_section` | lyrics refer to a section the plan does not have |
   | `duplicate_section` | lyrics list the same section twice |
   | `instrumental_section_has_lines` | lyrics on an instrumental section |
   | `syllable_range_inverted` | `line_syllables.min > max` |
   | `lyrics_too_long` | the lyrics' line text totals more than 20,000 characters (path `sections`) |
   | `document_too_large` | the normalized document is larger than 131,072 bytes as compact JSON (path `""`, the root) |
   | `too_long_after_normalization` | a value is longer than its documented limit once normalized (NFC can expand characters); path is the value's own, e.g. `title`, `sections[2].style.tags[1]`, `style_tags[0]` |
   | `requires_make_current` | `incompatible_lyrics: "detach"` was sent with `make_current: false` (path `incompatible_lyrics`); checked first, before anything else is read |

   The size codes and `too_long_after_normalization` are collected together with every other issue; they never short-circuit validation.

   `reserved_descriptor`, `reserved_delivery_tag` and `reserved_tempo_tag` apply to **new writes only**: versions and artists stored earlier that contain such values still load, render and export. (An import over a plan that still holds a tempo-like global tag, or on a project whose artist still holds a tempo-like style tag, is refused with the same code, §10; the export still works for both.)

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
- The current pair is always **compatible**: every lyrics section exists in the plan, none sits on an instrumental section, and — if the plan itself is `instrumental` — the lyrics have **no line at all** (`instrumental_plan_has_lines`, alongside the per-section `instrumental_section_has_lines`). Creating lyrics with lines for an instrumental plan is `422`; making such a plan current over lyrics that have lines is `409 lyrics_incompatible_with_plan` (or a detach, §5 table); `PUT …/current` and the export refuse it with the same `409`.
- A `409` here is **atomic** — nothing is written (no row, no number, no pointer).

| Action | Rule |
|---|---|
| Create **lyrics**, `make_current: true` | needs a current plan (`409 plan_required`); checked against the **current** plan, not only its base (`409 lyrics_incompatible_with_plan`, `detail.issues`). With `make_current: false` it is stored without checks against the current plan. |
| Create **plan**, `make_current: true` | if the current lyrics no longer fit: `incompatible_lyrics: "reject"` (default) → `409 lyrics_incompatible_with_plan`; `"detach"` → the plan becomes current, the **lyrics pointer is cleared** (the lyrics version itself is kept) and the response has `lyrics_detached: true`. With compatible lyrics `detach` changes nothing. `"detach"` **requires `make_current: true`**: with `make_current: false` there is no change to the pair, so the request is refused with `422 document_invalid` (`{path: "incompatible_lyrics", code: "requires_make_current"}`) and nothing is written. |
| `PUT …/current` | each given id must be a version of this track **of the matching kind** (`404` otherwise). Omitted fields keep their pointer; `lyrics_version_id: null` detaches the lyrics. The resulting pair must be compatible. To move to a plan with different sections send `{plan_version_id: X, lyrics_version_id: null}`. |

There is deliberately **no silent auto-detach**: dropping the lyrics is always the caller's explicit choice.

---

## 6. Deleting versions

`DELETE /versions/{id}` succeeds only if the version is **not current**, has **no children**, is **not the base of a lyrics version** and is **not the input of a run in progress**:

| Refusal | Code | `detail` |
|---|---|---|
| it is the track's current plan or lyrics | `409 version_is_current` | |
| other versions list it as `parent_version_id` | `409 version_has_children` | `{count}` |
| lyrics versions are based on this plan | `409 version_is_lyrics_base` | `{count}` |
| a `queued` or `running` run takes it as its plan or lyrics input (§11) | `409 version_in_use_by_run` | |

`version_in_use_by_run` clears by itself: the run ends (or is cancelled while `queued`) and the version can be deleted. A `lyrics` result is tied to the section ids of its plan, so a plan cannot disappear under a run that is writing lyrics for it.

Version numbers are never reused: after deleting #3, the next version is #4.

---

## 7. Other errors

Every error is the unified envelope `{error, message, status_code, detail}`. `message` is static English text — use `error` for logic.

| Status | `error` | When |
|---|---|---|
| 400 | `invalid_cursor` | malformed `cursor` |
| 404 | `not_found` | missing, someone else's, or another product's resource (never distinguishable) |
| 404 | `artist_not_found` | `artist_id` is not one of the caller's artists |
| 404 | `memory_not_found` | the memory is unknown, or belongs to another user, product or artist (never distinguishable, §12) |
| 404 | `plan_version_not_found` | a `lyrics` run's `plan_version_id` is not a plan version of this track (§11) |
| 404 | `lyrics_version_not_found` | a `revise_lyrics` run's `lyrics_version_id` is not a lyrics version of this track (§11) |
| 409 | `artist_name_conflict` | name already used (case-insensitive) |
| 409 | `artist_limit_reached` | 50 artists |
| 409 | `artist_in_use` | `detail: {project_count}` |
| 409 | `memory_limit_reached` | the artist already has 200 **active** memories (creating, or restoring an archived one, §12) |
| 409 | `project_has_tracks` | disabling studio on a project with tracks |
| 409 | `track_limit_reached` | 64 tracks |
| 409 | `plan_required` | lyrics made current without a plan; a `lyrics` run with no plan to write for (§11) |
| 409 | `lyrics_required` | a `revise_lyrics` run on a track with no lyrics (§11) |
| 409 | `version_in_use_by_run` | deleting the input of a run in progress (§6) |
| 422 | `track_order_mismatch` | `track_ids` is not exactly the project's track set (missing, extra or duplicate id) |
| 422 | `document_invalid` | §4 — also artist / track / project-profile / memory requests (empty or over-long once normalized, tag conflicts, …), and the pinned `memory_ids` of a run (§11, §12) |
| 422 | `import_invalid` | §10 — pasted Suno text that cannot become valid documents; issues carry the pasted line they come from |
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
  instrumental: boolean;             // the plan's track-level flag — see below
  fields: { title | style | exclude | lyrics: { length: number; limit: number | null; within_limit: boolean } };
  warnings: { code: string; section_id: UUID | null }[];
}
```

**Rendering rules**

- `style` = plan `global_style.tags`, then the project artist's `style_tags`, de-duplicated case-insensitively (first occurrence wins), then `"<tempo> BPM"` — at most one tempo item, always **last** (no tag can read as a tempo, `reserved_tempo_tag`). `exclude` = plan `global_style.avoid` then artist `global_avoid`, same de-duplication.
- `lyrics` = one block per plan section, in plan order, separated by a blank line. Header `[Label]` or `[Label: descriptors]` where descriptors are the section's `delivery` then its `style.tags`. Label = the section's `label`, else the kind's name (`Pre-Chorus`, `Post-Chorus`, otherwise title-case); a defaulted label is numbered (`Verse 1`, `Verse 2`) when the kind occurs more than once — except `chorus`, `pre_chorus`, `post_chorus` and `hook`, which stay unnumbered. A line with `delivery` is preceded by its own `[d1, d2]` line. **An instrumental section is never rendered without a cue:** a section with `instrumental: true` whose kind is not `instrumental` gets the descriptor `instrumental` **first** in its header (`[Drop: instrumental, heavy bass]`), de-duplicated case-insensitively against its own delivery and style tags, so a long tag list cannot push it out of sight; a section of kind `instrumental` is already headed `[Instrumental]` and gets no extra cue. **No parentheses are ever generated** (Suno sings them as backing vocals); `(…)` appears only if you typed it.
- **`instrumental`** is the plan's track-level flag, reported next to the text because Suno takes an "Instrumental" toggle rather than a tag: switch it on and paste `lyrics` as is. It adds **no** cue to any header (section cues stay driven by each section's own flag). A plan without sections renders `lyrics: ""` (length `0`, within the limit).
- **Never exported:** `references`, `idea`, section `notes`, `duration_seconds`, section-local `avoid`, the artist's `persona`, `voice_notes` and `aesthetic_references`.

**Limits are information, not errors.** Suno publishes no limits; the values below were measured on the live Suno v6 web UI and verified there on 2026-11-12. They may change. The export **never truncates and never fails on length** — the full text is returned and `within_limit: false` tells you to cut:

| Field | `limit` (verified on the live Suno v6 web UI, 2026-11-12) |
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
- **Open a track with one call:** `GET /tracks/{id}` returns the current plan and lyrics. The track and the two documents are read as **one snapshot**, so switching the current plan and deleting the old one in another tab never makes this call (or the Suno export of the current pair, or an import preview) fail: you get the pair that was current at that instant, and the next call shows the new one.
- **Branching history:** every version has `parent_version_id`; a "version tree" view is a straightforward render of `GET /tracks/{id}/versions?kind=…`. Deleting a leaf is allowed; deleting a node with children is not.
- **Optimistic UI is safe.** Because create is idempotent, a double submit yields one version.
- **Positions** can have gaps after a track delete; use `position` for ordering only and call the order endpoint to compact.
- **Artist PATCHes are serialized per artist.** Two overlapping `PATCH /artists/{id}` calls both succeed and neither loses the other's fields; the later one waits for the earlier one to commit and validates against the result (so a tag added by one and the same entry added to `global_avoid` by the other end as a `422 tag_conflict` for whichever runs second).
- **The import is the export's inverse** (§10): `export → paste into Suno → paste back` of an unchanged song is a no-op (`200`, nothing written). Offer "import from Suno" as a text area per box with a **Preview** (`dry_run: true`) before **Import**.
- The studio profile's `artist_id` change (via `PUT`) affects the *next* export immediately — exports always read the artist's current style.
- **Artist names are compared with `lower()`, not `casefold()`** (the same rule as library project names; PostgreSQL has no `casefold`). Plain case differences collide (`Nova` / `NOVA`, and non-ASCII letters such as `Рок` / `рок` under the default UTF-8 database locale), but a few characters that only casefold equates stay distinct: `Straße` and `STRASSE` are two valid names. Tags, labels and the other free-text lists use `casefold` (`duplicate_tag` catches `ß` / `SS`). This is a known, accepted deviation; do not rely on name uniqueness being casefold-strict.

---

## 10. Suno import

`POST /v1/studio/tracks/{id}/imports/suno` — paste Suno's three boxes back and the server reads them as the track's next plan and lyrics. **Deterministic** (no LLM; the same text over the same track always gives the same result), **free** (no billing), same guards as the rest of studio (`404` when off, `401`, `428` on a stale legal acceptance).

```ts
interface SunoImportRequest {        // unknown fields → 400
  title?: string | null;             // 1–200; plan title. Omitted/null: the CURRENT plan's title (a track with no plan: the track's title)
  style?: string;                    // ≤ 1000 ("" default)
  exclude?: string;                  // ≤ 1000
  lyrics?: string;                   // ≤ 20,000
  instrumental?: boolean | null;     // omitted/null: the CURRENT plan's flag (no plan: false). NEVER inferred from the text — send true/false to set it
  language?: string | null;          // lyrics language (same format as LyricsDocument.language)
  dry_run?: boolean;                 // false. Parse, validate, report; write nothing
}
interface SunoImportResult {
  dry_run: boolean;
  plan: PlanDocument;                // the NORMALIZED plan the text stands for (stored, or on a dry run, would be)
  lyrics: LyricsDocument | null;     // null when the Lyrics box has no lyric line
  plan_unchanged: boolean;           // equal to the current plan: no row written
  lyrics_unchanged: boolean;
  warnings: { code: string; line: number | null }[];
  round_trip: { style: boolean; exclude: boolean; lyrics: boolean };
  plan_version: VersionSummary | null;     // current after the import; null on a dry run
  lyrics_version: VersionSummary | null;   // null on a dry run, or when the song has no lyrics
  created: boolean;                  // any version row was written
}
```

**Status:** `201` when a version row was written, `200` otherwise — a dry run, or a paste that changes nothing. "Nothing changed" is never a `409`.

### What an import does

- A real import writes the plan version and, if the Lyrics box has lyric lines, the lyrics version, both with `origin: "import"`, **both made current**, in **one transaction**: if any step fails nothing is kept. They are written like typed versions (§4), so the same rules apply, and the new versions are **children of the track's current plan / lyrics**.
- **Unchanged** (`*_unchanged: true`): the result equals the current version → no row, no version number used. The lyrics version records the plan version it was written for, so lyrics are unchanged only when the plan is unchanged too: changing only Style (say) writes a new plan version **and** a new lyrics version with the same lines, based on the new plan.
- **A dry run** returns the same `plan`, `lyrics`, `warnings`, `round_trip` and `*_unchanged` as the real import would, and **writes nothing** — no row, no pointer, no version number — and takes no lock. Use it to preview.
- **No lyric lines** (`lyrics: null`): the plan keeps the pasted headers as sections, no lyrics version is written and, if the track had current lyrics, the **lyrics pointer is cleared** (the old lyrics versions are kept). A headers-only paste on a track that is not instrumental is reported as `no_lyric_lines`; an empty Lyrics box is simply a plan with no sections.

### How the text is read

- **Style / Exclude:** split on `,`, each item NFC-normalized and trimmed, empty items skipped, repeats removed case-insensitively (first wins, `duplicate_style_item`). In Style, the **first** item like `120 BPM` / `120bpm` (40–240, anywhere in the list) becomes `tempo_bpm`; **later tempo items are dropped** (`duplicate_tempo`) — a tag that reads as a tempo is reserved (`reserved_tempo_tag`, §4), so the plan could not hold it; anything out of range (`300 BPM`) is an ordinary tag. The export writes the tempo last, so for an unchanged song this is an exact inverse. The **project artist's tags are taken back out** — the exact inverse of the export's merge — so they are not copied into the plan; if the current plan's own tags already render to the pasted text they are kept as they are.
- **Lyrics:** lines are NFC-normalized and trimmed; `\r\n` and `\r` are line breaks. A **bracket line** (`[…]`) is a *section header* if it opens a blank-line-separated block, or if what it holds reads as a header (its text before the first `:` is a known section name, optionally followed by a number: `Verse`, `Verse 2`, `Pre-Chorus`, `Refrain`, `Fade Out`, `Break`, `Solo`, …). Any other bracket line is the **delivery of the next text line**. A text line belongs to the current section; a blank line then a text line (not a header) continues it.
- **Header → section.** `[Name: a, b]` splits at the **first** `:`; the descriptors are split on `,`. The name becomes the section's `kind` (via the alias table; an unknown name becomes `interlude` with the name kept as its label and the warning `unknown_section_name`). The `label` is `null` exactly when the name equals the header the export would write by default for that position (case-sensitive), otherwise the name is kept. A descriptor `instrumental` (any case, any position) sets the section's `instrumental` flag and is removed — a section of kind `instrumental` does **not** set it by itself. Every other descriptor becomes a section style tag (the delivery/tag split of the export is not recoverable from text).
- **Section ids (stable).** Each pasted section reuses the id of a not-yet-used section of the *current plan*, found in **two passes**, each taking the first unused match in order: (1) the section whose **whole rendered header line** equals the pasted one (so a label that itself contains `:`, like `Verse 1: The Return`, finds its section again, and an unchanged header reclaims its own section even when sections were reordered); then (2) for what is left, the section whose rendered header *name* equals the pasted name. A section with no counterpart gets the deterministic id `uuid5(track_id, "suno-import:<ordinal>:<header name>")` (ordinal = 0-based position among the pasted sections), so importing the same text twice gives the same ids. Lyrics therefore keep pointing at the right sections across edits.
- **What the text cannot say is carried over from the current plan/lyrics**, so `export → Suno → import` of an unchanged song changes nothing: a matched section whose **whole header line is identical** to the current plan's is copied verbatim (kind, label, the delivery/tag split, instrumental, local avoid list, planned duration, notes); any other matched section keeps its planned duration, notes and local `avoid`; the plan keeps its `idea` and `references`; the lyrics keep `line_syllables`. Pasting into a track with no plan creates everything from the text alone.
- **Language** (needed only when there are lyric lines): the request's `language`, else the current lyrics' language, else the project artist's `default_language`, else `422 language_required`.
- **Never silent, never guessed:** parentheses are kept as typed; `instrumental` is only what you send; every guess is a warning.

### Warnings

| Code | `line` | Meaning |
|---|---|---|
| `no_section_headers` | first text line | lyric lines but no header at all: read as one `verse` |
| `text_before_first_header` | first such line | text above the first header: read as an implicit `verse` |
| `dangling_delivery` | the bracket line | a delivery line with no text line after it (or replaced by a second one): dropped |
| `unknown_section_name` | the header | the header name is not a known section: kind `interlude`, label kept |
| `duplicate_descriptor` | the header / delivery line | a repeated descriptor (case-insensitive): collapsed |
| `no_lyric_lines` | — | headers only, no lyric line, and `instrumental` was not set |
| `duplicate_style_item` | — | a repeated Style / Exclude item (case-insensitive): dropped |
| `duplicate_tempo` | — | a second (or later) BPM item in Style: dropped; the first one is the tempo |

Every warning has a `code`; ones that come from the Lyrics box carry the **1-based line** of the pasted Lyrics, the others `null`.

### `round_trip`

For each box: after canonicalisation (NFC, `\n` line ends, lines trimmed, runs of blank lines collapsed and the ends trimmed; for Style / Exclude: split on `,`, trimmed, rejoined with `", "`), does **rendering the imported documents give back exactly what was pasted**? `false` is information, never an error: `120bpm` is written back as `120 BPM`, a tempo that was not the last Style item (`120 BPM, rock`) is written back last (`rock, 120 BPM`), a dropped dangling bracket is gone, a repeated descriptor (`[Verse: soft, Soft]`) or Style item is collapsed, an empty item (`a,,b`) is skipped. Show a "this will be normalized as…" hint for any `false`.

### `422 import_invalid`

```json
{ "error": "import_invalid",
  "detail": { "issues": [{ "path": "plan.sections[1].label", "code": "reserved_characters" }],
              "source_lines": { "plan.sections[1].label": 4 } } }
```

`issues` lists **every** problem; `source_lines` maps a path to the 1-based line of the pasted Lyrics it comes from (a number; a list of numbers for `instrumental_with_lyric_lines`). Neither ever contains pasted text. Nothing is written. Codes:

| Code | Path | Meaning |
|---|---|---|
| `unsupported_inline_tag` | `lyrics[<line>]` | a lyric line contains `[` or `]` but is not a whole bracket line (e.g. `la [soft] la`); one issue per such line |
| `instrumental_with_lyric_lines` | `lyrics` | the track is `instrumental` (the request says so, or — when it says nothing — the current plan does) and the Lyrics box has lyric lines; `source_lines.lyrics` lists them all. To write lyrics over an instrumental track send `instrumental: false` |
| `reserved_tempo_tag` | `plan.global_style.tags[i]`, `artist.style_tags[i]` | the **current plan** still holds a global tag that reads as a tempo, or the **project artist** still holds a style tag that reads as one (both stored before the tag was reserved). The import is refused rather than silently rewriting the plan (an artist tag would otherwise become, or override, the plan's tempo): save a plan version without that tag (or with it moved to `tempo_bpm`), or fix the artist (`PATCH /v1/studio/artists/{id}` with a `style_tags` that has no tempo), first. A plan and an artist that are both affected are reported together, the plan's paths first. The artist's `global_avoid` is not checked (nothing reads a tempo from Exclude) |
| `language_required` | `language` | there are lyric lines and no language could be determined |
| any §4 code | `plan.<path>` / `lyrics.<path>` | the imported plan / lyrics failed validation, e.g. `plan.sections[2].label` `reserved_characters`, `plan.global_style.avoid[0]` `tag_conflict`, `lyrics.sections[1]` `instrumental_section_has_lines` (a lyric line under `[Instrumental]`), `lyrics.sections[0].lines[3].text` `too_long_after_normalization`. Style / Exclude issues have no `source_lines` entry |
| `too_many_items` | `plan.global_style.tags`, `plan.sections`, `lyrics.sections[i].lines`, … | more items than the document allows (40 tags, 64 sections, 64 lines per section, …) |

Other statuses: `404` (not your track, any product mismatch, studio off), `400` (malformed body or a limit above), `428`, and the `409`s of §4/§5 for a version-create that cannot proceed (e.g. `version_limit_reached`) — in which case, as always, nothing is written.

### Things to know

- A label containing `:` is read as a name plus descriptors (`[Bridge: key: C]`) — the kind and the style tags come from that split — but the section still keeps its id and its label when the whole header line is unchanged (see "Section ids"). Only the part *before* the first `:` decides whether a bracket line inside a block is a header, so `[soft: verse]` in the middle of a block is a delivery. (A bracket line that **opens** a block is always a header, known name or not.)
- **What the request leaves out is the current plan's, not a default.** Omit `title` and `instrumental` (the usual paste of just the three boxes) and the plan keeps its own title and flag, so `export → paste back` of an unchanged song is a no-op even when the plan's title differs from the track's, or the plan is instrumental. A track with no plan yet uses the track's title and `instrumental: false`. The flag is still never inferred from the text: lyric lines pasted over an instrumental plan are `422 instrumental_with_lyric_lines` until you send `instrumental: false`.
- The import does not change the track's title, only the plan's.

---

## 11. Planner runs

A **run** is an asynchronous LLM job on a track. There are three kinds, one wire union on `kind`:

| `kind` | What it does | Result |
|---|---|---|
| `plan` | the user writes a short *brief*; the model proposes a song structure | a new **plan** version |
| `lyrics` | the model writes lyrics for a plan version | a new **lyrics** version |
| `revise_lyrics` | the model rewrites whole sections and/or single lines of existing lyrics, following an instruction | a new **lyrics** version |

In every case the server validates what the model returns **exactly as it would validate a document typed by the user**, and saves it as an **ordinary version** (`origin: "llm"`, never made current).

> **Privacy — tell the user.** What a run sends to the **configured LLM provider**: the track title, the project concept, the artist (name, persona, voice notes, style tags, things to avoid, aesthetic references) and the **artist memories the run uses** (title, date, salience, themes, imagery and body: §12); for `plan` the brief and a short summary of the track's current plan; for `lyrics` the **whole plan** (title, idea, tempo, tags, every section with its delivery, tags and notes), the optional brief and the avoid words; for `revise_lyrics` the plan, the **current lyrics**, the targets, the instruction and the avoid words. Nothing else leaves the server; a run is never traced to any third-party observability service. The server never logs any of it.

### Gating

Runs sit behind **their own switch**, because they spend money: `STUDIO_RUNS_ENABLED` (default off) **and** the studio gate of §1. While either is off, **all four** run routes answer `404 {"error": "not_found"}` (before authentication). A `404` on `POST …/runs` therefore means "planner runs are off" — hide the "Plan with AI" action; the rest of the studio keeps working. A user never chooses the model; the operator configures it: the shared model registry (`LLM_MODELS` / `LLM_DEFAULT_MODEL`, used by every feature that calls a model) and, per run kind, the studio's own routing (`STUDIO_LLM_MODEL_BY_KIND`).

### Endpoints

| Method | Path | Success | Notes |
|---|---|---|---|
| `POST` | `/v1/studio/tracks/{track_id}/runs` | `202` queued run · `200` replay | **`Idempotency-Key` header required** (1–128 chars) |
| `GET` | `/v1/studio/runs/{run_id}` | `200` run | the polling fallback of the event |
| `GET` | `/v1/studio/tracks/{track_id}/runs` | `200` `CursorPage<RunOut>` | newest first; `?limit=1..100` (default 30), `?cursor=` |
| `POST` | `/v1/studio/runs/{run_id}/cancel` | `200` run | only while `queued` |

All are subject to legal-acceptance enforcement (§1) for mutations, and are scoped to the caller: another user's track or run is `404`.

### Submit

```http
POST /v1/studio/tracks/{track_id}/runs
Idempotency-Key: 6f1c…           ← your own unique key per user action, 1–128 characters
Content-Type: application/json

{ "kind": "plan",
  "brief": "A slow synthwave song about driving home through the rain.",
  "language": "en",                ← optional BCP-47-ish tag, like lyrics
  "memory_ids": ["…"],             ← artist memories to pin, in the order the model should see them (§12); [] for none
  "auto_fill": true }               ← after the pins, fill the rest of the memory budget by salience
```

- `brief`: 1–4000 characters. It is NFC-normalized like every studio string, and the 4000 limit is re-checked **after** normalization (`422 document_invalid`, issue `brief` `too_long_after_normalization`; a blank brief is `empty_after_normalization`).
- `memory_ids` (≤ 50) and `auto_fill` exist on **every** kind and mean the same everywhere; see "Artist memories in a run" below and §12.
- Unknown fields are refused (`400`) in every kind. `kind` is the discriminant: `plan`, `lyrics` or `revise_lyrics`.
- The response is a **`RunOut`** (below) with `status: "queued"`, `revision: 1`. `202` means "accepted"; the work happens in the background.

#### `kind: "lyrics"` — write lyrics for a plan

```jsonc
{ "kind": "lyrics",
  "plan_version_id": "…",          // optional; default: the track's current plan
  "language": "en",                // optional; see "Language" below
  "line_syllables": { "min": 6, "max": 10 },   // optional; every line should have min..max syllables
  "avoid_words": ["baby"],         // optional, ≤ 30 items of 1–40 characters
  "brief": "about leaving a small town",       // optional extra direction, ≤ 4000 characters
  "memory_ids": [], "auto_fill": true }
```

- `avoid_words` are normalized like tags (NFC, trimmed, a repeat or an empty item is `422`, issue `avoid_words[i]`; square brackets are refused). They match **whole words or phrases, case-insensitively**, with no stemming: `baby` matches "Oh BABY, come" and not "babysitter".
- `line_syllables.min > max` is `422` (`line_syllables` / `syllable_range_inverted`).
- The run writes lines for **every section of the plan that is not instrumental**; instrumental sections get none.

#### `kind: "revise_lyrics"` — rewrite parts of lyrics

```jsonc
{ "kind": "revise_lyrics",
  "instruction": "make the chorus more hopeful",     // required, 1–1000 characters
  "targets": {
    "sections": ["<plan section id>"],               // rewrite the whole section (its number of lines may change)
    "lines": [{ "section_id": "<plan section id>", "line": 3 }]   // replace one line, 0-based
  },
  "lyrics_version_id": "…",        // optional; default: the track's current lyrics
  "line_syllables": { "min": 6, "max": 10 },         // optional; default: the input lyrics' own range
  "avoid_words": [], "memory_ids": [], "auto_fill": true }
```

- `targets` must name **something** (`422`, `targets` / `targets_required`), without repeats (`duplicate_target`) and without a line whose section is also a section target (`overlapping_targets`). Paths are `targets.sections[i]` / `targets.lines[i]`.
- **Everything not targeted stays byte-identical** to the input lyrics. A section target **may** name a vocal section that has no lines yet: the revision writes it. A line target never changes the number of lines.
- The revision starts from the **input lyrics' own plan** (`based_on_plan_version_id` of that lyrics version), which need not be the track's current plan. The language is the lyrics' own.

#### What a run works on is fixed when it is accepted

`plan_version_id` / `lyrics_version_id` / the language are **resolved at submit**, under the track lock, and stored on the run. A wrong or missing input is a `4xx` **with no run and no charge**. After that the run never looks at the track's current pointers again: switching the current plan or lyrics while the run works changes nothing about what it does. An idempotent replay returns the same run even after the pointers moved.

A version that a **queued or running** run takes as input cannot be deleted (`409 version_in_use_by_run`, §6).

| Check at submit | Error |
|---|---|
| `lyrics`: `plan_version_id` is not a plan version of this track | `404 plan_version_not_found` |
| `lyrics`: no `plan_version_id` and the track has no current plan | `409 plan_required` |
| `lyrics`: the plan is instrumental, or has no section that takes lines | `422 document_invalid` — `plan_version_id` / `plan_has_no_vocal_sections` |
| `lyrics`: no language (see below) | `422 document_invalid` — `language` / `language_required` |
| `revise_lyrics`: `lyrics_version_id` is not a lyrics version of this track | `404 lyrics_version_not_found` |
| `revise_lyrics`: no `lyrics_version_id` and the track has no current lyrics | `409 lyrics_required` |
| `revise_lyrics`: the input lyrics are **not valid today** against their plan | `422 document_invalid` — the §4 issues, paths prefixed `lyrics.` (fix the lyrics first) |
| `revise_lyrics`: a section target is not a section of the lyrics' plan | `422` — `targets.sections[i]` / `unknown_section` |
| `revise_lyrics`: a section target is an instrumental section | `422` — `targets.sections[i]` / `instrumental_section` |
| `revise_lyrics`: a line target's section has no lines, or `line` is past its last line | `422` — `targets.lines[i]` / `line_out_of_range` |

All target problems are reported together in one `422`.

**Language** (`lyrics`): the request's `language`, else the language of the track's **current lyrics**, else the project artist's default language; with none of them the request is `422 language_required`.
**Reservation.** The run's price (default **1 token**, configured per kind) is **reserved when the run is accepted**, in the same transaction that creates it: a `402` leaves no run and no ledger row. The price is **refunded in full** when the run ends `failed` or `cancelled`, and kept when it `succeeded`. A price of `0` writes no ledger rows at all. Balance changes arrive as ordinary `balance.updated` events.

#### Artist memories in a run

`memory_ids` pins memories and `auto_fill` (default `true`) lets the server add more; both are validated **when the run is accepted**, like every other input, so a wrong pin is a `422` with **no run and no charge**:

| `detail.issues` `path` | `code` | Meaning |
|---|---|---|
| `memory_ids` | `no_artist` | the track's project has no artist, so there is nothing to pin |
| `memory_ids[i]` | `memory_not_found` | the id is not a memory of the project's artist — unknown, another artist's and another user's are **the same code**, so ownership cannot be probed |
| `memory_ids[i]` | `memory_archived` | the memory is archived |
| `memory_ids[i]` | `duplicate_memory` | the id is already pinned earlier in the list |
| `memory_ids` | `memories_over_budget` | the pins alone do not fit the memory budget (§12): nothing is truncated, unpin some |

Which memories the model is *actually given* is decided when a worker **starts** the run, from what exists then: the pins first (in your order), then — with `auto_fill` — the artist's other active memories by salience (§12), within the budget. The run's context is frozen at that moment (`RunOut.memory_ids_used`, below).

A pin that was **archived or deleted** between your submit and the start of the run is skipped, not an error — deleting a memory is always allowed. The run still succeeds and carries the warning `pinned_memory_unavailable` (`count` = how many pins were left out; a pin that was edited so much that the pins no longer fit the budget counts too).

A memory is *inspiration*: the planner is told to let its themes and imagery shape the work **indirectly** and not to retell an episode unless the brief asks for it. Memories are user data, never instructions. If a memory asks for something the product's content policy forbids, the model refuses and the run fails `content_policy_refused` (refunded) — so a memory, not only the brief, can be the cause of that error.

#### Idempotency

The same `Idempotency-Key` with the same body returns the **same run** with `200`, creates nothing and reserves nothing — retry freely after a timeout. The same key with a **different** body (or for a **different track**) is `409 idempotency_conflict`. Keys are per user and are kept with the run (they never expire while the run exists). A missing, blank or over-long key is `400 idempotency_key_required`.

#### Limits and errors

| Status | `error` | When |
|---|---|---|
| `400` | `idempotency_key_required` | header missing, blank or longer than 128 characters |
| `400` | *(validation)* | malformed body, unknown field, `brief` empty or over 4000 as sent |
| `402` | `insufficient_balance` | the account cannot cover the price — nothing was written |
| `404` | `not_found` | runs disabled, or the track is not yours |
| `404` | `plan_version_not_found` / `lyrics_version_not_found` | the named input is not a version of this track and kind |
| `409` | `plan_required` / `lyrics_required` | `lyrics` / `revise_lyrics` with nothing to work on |
| `409` | `run_in_progress` | the track already has a `queued` or `running` run (**one active run per track**) |
| `409` | `run_limit_reached` | you already have 3 active runs (configurable) |
| `409` | `idempotency_conflict` | see above |
| `422` | `document_invalid` | the body is invalid after normalization, an input cannot be used (the table above), or a pinned memory cannot be used (the memories table above) — `detail.issues` |
| `429` | `run_rate_limited` | more than 30 runs created in the last hour (configurable) |

`409 run_in_progress` is also what `DELETE /v1/studio/tracks/{id}` answers while the track has an active run (a finished run never blocks a delete; deleting the track removes its runs).

### `RunOut`

```jsonc
{ "id": "…", "track_id": "…",
  "kind": "plan | lyrics | revise_lyrics",
  "status": "queued | running | succeeded | failed | cancelled",
  "stage": null,                      // running only: "context" | "drafting" | "validating" | "revising" | "saving"
  "revision": 1,                      // strictly increases with every status or stage change
  "price_tokens": 1, "model_key": "grok",   // the operator's `LLM_MODELS` key, informational
  "result_plan_version_id": null,     // set when a plan run succeeded
  "result_lyrics_version_id": null,   // set when a lyrics / revise_lyrics run succeeded
  "warnings": [],                     // [{ "code", "path", "count" }] (path / count null when they do not apply)
  "memory_ids_used": [],              // artist memories the model was given, in prompt order — ids only
  "error_code": null,                 // set when failed (table below)
  "error_detail": null,               // failed: issue paths and codes only — never model or user text
  "llm_calls": 0,                     // model calls made (plan ≤ 3, lyrics ≤ 5, revise_lyrics ≤ 3)
  "created_at": "…", "started_at": null, "finished_at": null }
```

`memory_ids_used` is empty until the run starts (the selection is made then) and for a run without memories; fetch the texts with `GET /v1/studio/memories/{id}` (a deleted memory is gone, but the run's own copy is untouched). It never contains the brief, the artist, any memory's text, the prompts, the model's output or token counts.

### Lifecycle

```
queued ──claim──▶ running ──▶ succeeded
   │                 └────▶ failed
   └──cancel──▶ cancelled
```

- A run is **terminal once**: `succeeded`, `failed` and `cancelled` never change again. `finished_at` is set exactly when the run is terminal.
- `running` carries a `stage`: `context` (the worker froze what the model will be told) → `drafting` (first model call) → `validating` → `revising` ↔ `validating` (at most twice) → `saving`. Stages are informational; `status` is authoritative.
- **Budget:** a `plan` run makes at most **3** model calls (the first draft and two revisions), a `lyrics` run at most **5** (the draft and three fix rounds), a `revise_lyrics` run at most **3** (the revision and two fix rounds). A reply that is not valid JSON for the schema, or that answers lines nobody asked about, is re-asked inside the same budget. A fix round asks the model for replacements of exactly the lines that were rejected, one for one — every accepted line is frozen — or, when something structural is wrong (a missing section), for the whole reply again. The whole run has a **180 s** deadline; a run whose worker died is failed `run_timeout` and refunded — it is never executed twice.
- **Cancel** works only while `queued`: `POST …/cancel` → `200` with `status: "cancelled"` and a full refund. A `running` or finished run answers `409 run_not_cancellable` (no free spend by start-and-cancel). Another user's run is `404`.

#### `error_code` of a `failed` run

| `error_code` | Meaning | `error_detail` |
|---|---|---|
| `llm_unavailable` | the model provider could not be reached or kept failing (after one retry) | — |
| `llm_rejected` | the provider refused the request | `{ "status_code": 4xx }` |
| `llm_output_truncated` | the reply hit the output limit | — |
| `llm_output_invalid` | the reply was not a valid object on the last allowed call | — |
| `content_policy_refused` | the model declined the brief **or one of the artist memories it was given** under the product's content policy | — |
| `brief_unusable` | the model found the brief (or, for a revision, the instruction) too empty, unclear or contradictory to act on | — |
| `plan_invalid` | after the allowed revisions the plan still failed validation | `{ "issues": [{ "path", "code" }] }` — the §4 vocabulary, with paths in the **plan document** |
| `lyrics_invalid` | after the allowed rounds the lyrics still failed a **hard** check (a bracket in a line, a delivery that reads as a section header, a line over 300 characters, a section over 64 lines, a missing section ...). Soft misses never cause this | `{ "issues": [{ "path", "code" }] }` — paths in the **input plan's sections**: `sections[p]` or `sections[p].lines[j]`, `p` = the section's index in the plan, `j` = the 0-based line. No lyrics document was saved, so these are the only paths that mean something |
| `run_timeout` | the 180 s deadline passed, or the worker died | — |
| *any §4–§6 code* | the save step was refused, e.g. `version_limit_reached` | — |
| `internal_error` | anything unexpected (logged server-side, with no text) | — |

All failures refund the reservation. A failed or cancelled run leaves the track exactly as it was.

### Result: an ordinary version, never made current

A `succeeded` `lyrics` or `revise_lyrics` run has `result_lyrics_version_id` instead; see "Lyrics results" below. A `succeeded` `plan` run has `result_plan_version_id`. That version is a plain plan version of §2.2 with `origin: "llm"`, `is_current: false` and `parent_version_id` = the plan that was **current when the run started** (`null` if the track had none) — unless that plan was deleted while the run was working: see "The parent rule" below. **The run never changes the track's current plan or lyrics** — the user looks at the result (`GET /v1/studio/versions/{id}`, or the version list) and adopts it with `PUT /v1/studio/tracks/{id}/current` (or `POST …/versions` with `make_current`), subject to the usual pair-compatibility rules (§5). If the model's plan is identical to the plan it started from, the result *is* that parent version. Deleting the result version later keeps the run and clears `result_plan_version_id`.

#### The parent rule

A run takes a few seconds to a minute and a half, and the user may keep working on the track meanwhile — including switching the current plan and deleting the old one (§5, §6). A result the user has already paid for is never thrown away because of that:

1. The result is a child of the plan that was current when the run **started**.
2. **If that version no longer exists when the result is saved**, the result becomes a child of the track's **current plan at that moment** (a root version, `parent_version_id: null`, when the track has none), and the run carries the warning `parent_version_replaced`. The run is still `succeeded` and still charged.
3. A plan that was only *switched away from* (not deleted) is still the parent: the lineage is the plan the run started from.
4. If the model's plan equals the parent that ends up being used, the result *is* that parent (as above).

#### Lyrics results

A `lyrics` or `revise_lyrics` result is a plain **lyrics** version of §2.2 (`origin: "llm"`, `is_current: false`, language and `line_syllables` as the run used them):

| Kind | `based_on_plan_version_id` | `parent_version_id` |
|---|---|---|
| `lyrics` | the plan the run wrote for (the one resolved at submit) | the lyrics that were **current when the run started** (`null` if none) — if those were deleted meanwhile, the track's current lyrics at save time (or `null`) and the warning `parent_version_replaced`, exactly like a plan run |
| `revise_lyrics` | the **input lyrics' plan** | the **input lyrics** (they cannot be deleted while the run is active) |

A revision that changes nothing **succeeds with the input version as its result** (`result_lyrics_version_id` = the version you revised, no new version is written) and the warning `revision_unchanged`. A `lyrics` result equal to its parent *is* that parent.

The run's lyrics are validated like any lyrics you could post (§4) — and then checked **softly** against what you asked for. After the model's call budget is spent:

* a **hard** problem fails the run (`lyrics_invalid`, refunded, no version);
* a **soft** miss does **not**: the run succeeds and the misses are listed as warnings, because one line you can fix by hand is not worth throwing away a paid, usable draft. The soft checks are *the syllable range* and *the avoid words*; a revision checks **only the lines it changed**, never your own.

The same holds when a later **fix round fails**. Once the model has produced a document that validates, the rounds that follow only try to improve it. If one of them fails at the model provider (unreachable, refused, truncated, or no usable reply within the call budget) while only soft misses were left, the run **keeps the valid document**: it `succeeded`, is charged, and carries the soft warnings plus `revision_incomplete`. If a **hard** problem was still open there is no valid document to keep, so the run fails with the provider's own `error_code` (`llm_unavailable`, `llm_rejected`, `llm_output_truncated` or `llm_output_invalid`) and is refunded. A model that *declines* in a fix round is still a refusal (`content_policy_refused` / `brief_unusable`), not a success.

##### Syllable counting

Counting depends on the language and is a heuristic, which is why a miss is only ever a warning. Russian, Ukrainian and Bulgarian count one syllable per vowel letter; Serbian, Croatian and Bosnian (Cyrillic or Latin) count vowel letters plus a syllabic *r*; English counts vowel groups with a **±1 tolerance**. A line with a digit, or with a word outside the language's alphabet, is **not counted** (it is skipped and reported as a count). Any other language is not checked at all. The range you asked for is still stored on the result.

#### Run warnings

`warnings` lists things worth knowing about a **successful** run; each item is `{ "code": string, "path": string | null, "count": number | null }` (`path` and `count` are `null` when they do not apply), with no user text. A run with none has `[]`. At most 100 are kept: a longer list ends with `warnings_truncated` — except that `pinned_memory_unavailable`, when it applies, is always kept and comes first, and `revision_incomplete`, when it applies, is always kept and comes last (after `warnings_truncated`).

| `code` | Meaning |
|---|---|
| `parent_version_replaced` | the version the run started from (the plan, or for `lyrics` the current lyrics) was deleted before the result was saved; `parent_version_id` of the result is the track's current version of that kind at save time, or `null` (see "The parent rule"). Offer the user the result as usual; do not treat it as a failure. |
| `pinned_memory_unavailable` | `count`: pinned memories that were archived or deleted after you submitted (or no longer fit the budget) and so were **not** used. Any run kind; always the **first** warning when present. The run is otherwise complete |
| `revision_unchanged` | the revision changed nothing; the result is the version you revised |
| `revision_incomplete` | the run kept its best valid result because a later fix round failed; some soft warnings may remain. The result is complete and usable; the other warnings say which lines the run could not improve. Always the **last** warning of its run |
| `syllable_range_missed` | `path`: the line; `count`: the syllables counted (compare with `line_syllables`; English has ±1 tolerance already applied) |
| `avoid_word_used` | `path`: the line that contains an avoid word or phrase (the word itself is not reported) |
| `syllable_check_skipped_lines` | `count`: lines that could not be counted (a digit or a word outside the language's alphabet) |
| `syllable_check_skipped` | a `line_syllables` range was asked for, but the language has no counter |
| `warnings_truncated` | there were more warnings than are kept |

**Known limit.** The 180 s deadline is not a port failure: if it cancels a run **during a fix round**, the run still ends `failed` / `run_timeout` (and is refunded) even though a valid document existed. Recovering state from a cancelled run is deliberately not attempted.

**Paths.** The `path` of a warning on a **succeeded** run indexes the **result lyrics document**: `sections[i].lines[j]`, both 0-based — exactly the document `GET /v1/studio/versions/{result_lyrics_version_id}` returns. (For a `revision_unchanged` result that is the input lyrics.) The `issues` of a **failed** run's `error_detail` index the **input plan's sections**, as described above. The two are different index spaces on purpose: a failed run saved nothing, so the plan is the only structure the client has.

The plan the model returns is validated like user input (§4): normalization, the reserved descriptor / tempo-tag rules, caps and limits. Section ids are minted by the server. `idea` holds the brief (so it is stored with the plan and **never exported** to Suno); `references` stay empty.

### Event: `studio.run.updated`

Published on the user's SSE channel after **every committed status or stage change** (submit, claim, each stage, finish, cancel, sweep):

```json
{ "event_type": "studio.run.updated",
  "payload": { "run_id": "…", "track_id": "…", "kind": "plan",
               "status": "running", "stage": "drafting", "revision": 3, "error_code": null } }
```

- Delivery is **lossy** like every event: apply a frame only when its `revision` is **strictly greater** than the one you hold for that `run_id`; otherwise ignore it. On connect / reconnect, and whenever you doubt, `GET /v1/studio/runs/{id}` — REST is the complete fallback.
- The event carries ids and codes only. Fetch the run for the rest, and the version for the result.
- No text is streamed: a run is a short job and its result arrives whole, as a version.

### Notes for the UI

- Treat `202` and `200` the same way ("here is the run"); generate a fresh `Idempotency-Key` per *user action* (not per retry).
- While a track's run is `queued` or `running`, disable "Plan with AI" for that track (`run_in_progress` is the server-side guard) and show `stage`.
- A `failed` run has been refunded already — say so. Offer "try again" with a new key.
- Show the privacy note (above) next to the brief / instruction field, and the price (`price_tokens`) before submitting.
- For `revise_lyrics`, let the user pick targets from the lyrics they are looking at (section ids from the plan, 0-based line numbers). If the input lyrics are no longer valid (`422` with `lyrics.`-prefixed issues), send the user to fix them first.
- After a `lyrics` / `revise_lyrics` success, show the warnings on the lines they point at, and offer the result as a new version — never switch the current lyrics for the user.
- A result with `revision_incomplete` is a complete, valid version: offer it like any other, show the soft warnings that remain, and let the user run again if they want those lines improved. A `lyrics_invalid` or `run_timeout` failure never has a result to offer, whatever the run did before it failed.

---

## 12. Artist memories

A **memory** is one episode of an artist's (fictional) life — a title, a body, themes, concrete imagery, a salience. Planner runs (§11) read them as *inspiration*, so lyrics and plans carry the artist's history without retelling it. Memories belong to the **artist**, not to a project: every project that uses the artist sees the same ones. They never reach the Suno export (§8): they are planner-only.

> **Privacy — tell the user.** A memory's text (title, date, salience, themes, imagery, body) is sent to the **configured LLM provider** when a run uses it, and nowhere else. The server never logs it, a run's response carries only ids, and nothing is traced to any third-party service.

### Fields and limits

| Field | Limit | Notes |
|---|---|---|
| `title` | 1–120 | one line: whitespace (line breaks included) is collapsed to single spaces |
| `body` | 1–2000 | NFC, trimmed; line breaks and blank lines are kept |
| `occurred_on` | date or `null` | a **fictional** date, `YYYY-MM-DD`; ordering and display only |
| `themes` | ≤ 12 items × ≤ 60 | trimmed; an empty item, a case-insensitive repeat (`duplicate_tag`) or a square bracket (`reserved_characters`) is refused |
| `imagery` | ≤ 20 items × ≤ 120 | same rules as `themes` |
| `salience` | 1–5, default 3 | how readily the planner reaches for it when it fills the budget itself |
| `status` | `active` \| `archived` | an archived memory is kept but never used by a run; **new memories are always `active`** |
| `origin` | `user` \| `chat` | read-only; `user` today. Sending it is a `400` |

Every length is **re-checked after normalization** (Unicode NFC can make a value longer than it looked): `422 document_invalid`, issue `title` / `body` / `themes[i]` / `imagery[i]` with `too_long_after_normalization`. All issues of one request are reported together, at the wire path of the value. Nothing is ever truncated.

### Endpoints

All under `/v1/studio` (§1: `404` when studio is off; legal-acceptance enforced on mutations).

| Method | Path | Success |
|---|---|---|
| `GET` | `/artists/{artist_id}/memories` | `200` page, newest first (`created_at DESC, id DESC`). `?status=active\|archived` (default `active`), `?limit=1..50` (30), `?cursor=` |
| `POST` | `/artists/{artist_id}/memories` | `201` `Memory` |
| `GET` | `/memories/{memory_id}` | `200` `Memory` |
| `PATCH` | `/memories/{memory_id}` | `200` `Memory` — omitted fields are kept, `occurred_on: null` clears the date, `status` archives or restores |
| `DELETE` | `/memories/{memory_id}` | `204` |

The list is **not** the selection order (below): it pages by creation time so that editing a salience never moves a memory between pages.

Errors: `404 artist_not_found` (list, create), `404 memory_not_found` (the other three; also another user's, another product's or an unknown id, indistinguishably), `422 document_invalid`, `409 memory_limit_reached`, `400 invalid_cursor`.

### The 200 cap

An artist holds at most **200 active** memories. Creating the 201st, or **restoring** an archived memory when 200 are active, is `409 memory_limit_reached`. The cap is exact even under concurrent requests: archive some memories (or delete them) to make room. Archived memories do not count and have no limit of their own.

### Delete any time

A memory can be edited, archived or deleted **at any moment**, including while a run that uses it is queued or running. There is no `409` on delete. A run keeps its own copy of the text it was given, so what you do to a memory afterwards never changes a run that has started, nor its result.

### How a run chooses memories

1. **Pins first**, in the order of `memory_ids`.
2. With `auto_fill` (default `true`): the artist's other **active** memories in a fixed order — `salience` high to low, then `occurred_on` newest first (memories without a date last), then by id.
3. Taken **in that order while the next memory fits the budget**; the **first one that does not fit ends the selection** (a smaller, less salient memory behind it is *not* taken in its place — that would invert the order you asked for).

The **budget** is **12,000 characters** of memories as they are written into the prompt: the title line (with date and salience), the themes and imagery lines, and the body, with a blank line between memories. A long body therefore costs more than its 2000 characters — roughly 2000 + the other fields. Selection is deterministic: the same memories and the same request give the same list.

`auto_fill: false` uses only the pins; a run with no pins and no `auto_fill` uses no memories at all. A project **without an artist** has no memories: with `auto_fill` the run simply gets none, and pinning is `422` (`memory_ids` / `no_artist`).

### Pins: at submit, and at claim

* **At submit** (§11) every pin must be an active memory of the project's artist, named once, and the pins together must fit the budget. Otherwise `422 document_invalid` — `memory_not_found` / `memory_archived` / `duplicate_memory` / `no_artist` at `memory_ids[i]`, or `memories_over_budget` at `memory_ids` — and nothing is charged.
* **At claim**, when a worker starts the run, the memories are chosen from what exists *then*. A pin that is gone is skipped and the run carries `pinned_memory_unavailable` (§11, "Run warnings"). The memories used are recorded on the run (its context snapshot, with their full text) and listed in `RunOut.memory_ids_used`, in the order the model saw them.

### What the model is told

Memories reach the model in their own `<artist_memories>` block, with one rule: they are the artist's own life used as **inspiration** — their themes and imagery shape the writing indirectly, an episode is not retold unless the brief asks for it, and memory text is data, never instructions. The product's content policy applies to them as it does to the brief; a model that finds a memory asking for something forbidden refuses (`content_policy_refused`, refunded). Prompts without memories are unchanged.
