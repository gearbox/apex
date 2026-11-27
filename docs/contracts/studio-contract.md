# Frontend Contract — Media Project Studio (S1-P1 / S1-P1b / S1-P2a / S1-P2b / S1-P2c / S1-P3 / S2a / S3a: Studio Foundation, Suno import, Planner and lyrics runs, Artist memories, Artist chat, Cast, Track audio, Timelines)

> **Audience:** `gearbox/apex-frontend` (artists, studio projects, track editor, version history, Suno export and import).
> **Backend source:** branches `feat/studio-foundation` (migration `055_studio_foundation`), `feat/studio-suno-import` (migration `056_studio_plan_schema_v2`, plan **schema v2**) `feat/studio-planner-runs` (migration `057_studio_runs`), `feat/studio-lyrics-runs` (migration `058_studio_run_inputs`), `feat/studio-artist-memories` (migration `059_studio_artist_memories`) and `feat/studio-artist-chat` (migration `060_studio_artist_chat`) and `feat/studio-cast` (migration `061_studio_cast_asset_holds`) and `feat/studio-track-audio` (migration `062_studio_track_audio`) and `feat/studio-timelines` (migration `063_studio_audio_timelines`).
> **Authority:** `gen:api` (OpenAPI `schema.json`) is authoritative for **types**. This document is authoritative for **semantics**: limits, normalization, error codes, idempotency, the "current pair" rules and what the Suno export does and does not contain.
> **Scope of this phase:** artists, a studio profile on a library project, tracks, versioned Plan / Lyrics documents, a deterministic Suno export and its inverse, a deterministic Suno-text import (§10). and — behind a separate switch — LLM **planner runs** that draft a plan, write lyrics or revise them (§11), grounded in an artist's **memories** (§12) — and, behind its own switch, an **artist chat** (§13): "interview your artist", whose answers you can turn into new memories. and a **cast** (§14): people and characters with labelled reference images that the library keeps for as long as they are used. and **track audio** (§15): your own finished tracks, uploaded as audio files and attached to a track as numbered versions, and **timelines** (§16): where each section and lyric line happens in an audio version, with LRC / SRT lyric files. **No audio generation, no automatic alignment, no fan channels** — those arrive in later phases and will extend, not break, this contract.

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
- You can **interview an artist** (§13): chats, with replies that stream in over the same SSE stream as everything else, a few tokens per message. Nothing said in a chat is canon until you save it: "turn this into a memory" drafts a memory from any message, and you edit and save it (`origin: "chat"`).
- You can keep a **cast** (§14): up to 50 members per user, each with up to 12 labelled reference images from your library and up to 12 named looks. A reference **pins** its library asset: it is not swept by retention and cannot be deleted while a reference uses it (`409 asset_held`).
- You can attach **your own audio** to a track (§15): upload the file to the library, then attach it as a numbered, labelled **audio version** (up to 20 per track). Like a cast reference it **pins** its library asset. Nothing costs tokens.
- You can **time** an audio version (§16): mark when each plan section and, optionally, each lyric line happens, save it as immutable **timeline** versions, and export **LRC / SRT**. The server never invents a time. Free.
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
  performer_cast_member_id: UUID | null; // the cast member who performs this artist (§14); null = none
  created_at: ISODateTime;
  updated_at: ISODateTime;
}
interface ArtistCreate {             // unknown fields → 400
  name: string;
  persona?: string | null; voice_notes?: string | null; default_language?: string | null;
  aesthetic_references?: string[]; style_tags?: string[]; global_avoid?: string[];
  performer_cast_member_id?: UUID | null;   // must be one of the caller's cast members, else 404 cast_member_not_found
}
type ArtistPatch = Partial<ArtistCreate>;   // omitted = keep; null clears persona/voice_notes/default_language/performer_cast_member_id

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
  current_audio_version_id: UUID | null;    // §15; independent of the plan / lyrics pair
  created_at: ISODateTime; updated_at: ISODateTime;
}
interface TrackDetail extends Track {
  current_plan: PlanVersion | null;
  current_lyrics: LyricsVersion | null;
  current_audio: TrackAudio | null;        // §15
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
| DELETE | `/artists/{id}` | 204 — its memories, **chats, messages and memory drafts** go with it | `409 artist_busy` while a chat reply or a memory draft of the artist is `pending` (§13); else `409 artist_in_use` while a project uses it. When both apply, `artist_busy` is the answer |
| GET | `/artists/{id}/memories` | 200 page of `Memory` | `?status=active\|archived` (default `active`), `limit` 1–50 (30), diary order — see §12 |
| POST | `/artists/{id}/memories` | 201 `Memory` | max **200 active** per artist; `409 memory_limit_reached` — see §12 |
| GET / PATCH | `/memories/{id}` | 200 `Memory` | PATCH also archives / restores (`status`) |
| DELETE | `/memories/{id}` | 204 | always allowed, even while a run uses it — see §12 |
| GET / POST / … | `/cast…` | see §14 | the cast: members, references, looks |
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
| PUT | `/tracks/{id}/current` | 200 `Track` | body `{plan_version_id?, lyrics_version_id?, audio_version_id?}` — see §5 and §15 |
| GET / POST | `/tracks/{id}/audio` | 200 page of `TrackAudio` / **201** `TrackAudio` | the track's audio versions, newest first / attach an uploaded audio file — see §15 |
| GET / PATCH / DELETE | `/audio/{id}` | 200 / 200 / 204 | see §15; deleting an audio version deletes its timelines (§16) |
| GET / POST | `/audio/{id}/timelines` | 200 page of `TimelineSummary` / **201** created / **200** replay (`Timeline`) | see §16 |
| GET / DELETE | `/timelines/{id}` | 200 `Timeline` / 204 | see §16 |
| PUT | `/audio/{id}/current-timeline` | 200 `TrackAudio` | body `{timeline_id}` (`null` clears), §16 |
| GET | `/timelines/{id}/exports/{lrc\|srt}` | 200 `TimelineExport` | see §16 |
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

**The audio pointer is independent of the pair.** `PUT …/current` also takes `audio_version_id`: omitted keeps the pointer, `null` clears it, an audio version of **this** track sets it (another track's id, or an unknown one, is `404 audio_version_not_found`). Changing it never touches, and is never checked against, the current plan and lyrics. See §15.

---

## 6. Deleting versions

`DELETE /versions/{id}` succeeds only if the version is **not current**, has **no children**, is **not the base of a lyrics version**, is **not realized by an audio version** and is **not the input of a run in progress**:

| Refusal | Code | `detail` |
|---|---|---|
| it is the track's current plan or lyrics | `409 version_is_current` | |
| other versions list it as `parent_version_id` | `409 version_has_children` | `{count}` |
| lyrics versions are based on this plan | `409 version_is_lyrics_base` | `{count}` |
| audio versions of the track name it as the plan or lyrics they realize (§15) | `409 version_has_audio` | `{count}` |
| a `queued` or `running` run takes it as its plan or lyrics input (§11) | `409 version_in_use_by_run` | |

`version_in_use_by_run` clears by itself: the run ends (or is cancelled while `queued`) and the version can be deleted. A `lyrics` result is tied to the section ids of its plan, so a plan cannot disappear under a run that is writing lyrics for it.

Version numbers are never reused: after deleting #3, the next version is #4.

---

## 7. Other errors

Every error is the unified envelope `{error, message, status_code, detail}`. `message` is static English text — use `error` for logic.

| Status | `error` | When |
|---|---|---|
| 400 | `invalid_cursor` | malformed `cursor` |
| 400 | `invalid_character` | U+0000 anywhere in a JSON body, query string or path (any route, not only studio); `detail: {location: "body" \| "query" \| "path"}`. The frontend should never send NUL: this is a client bug, not a user error |
| 400 | `invalid_asset_ref` | `?holding=` is not an asset ref (`upload:<uuid>` / `output:<uuid>`), §14 |
| 404 | `not_found` | missing, someone else's, or another product's resource (never distinguishable) |
| 404 | `artist_not_found` | `artist_id` is not one of the caller's artists |
| 404 | `memory_not_found` | the memory is unknown, or belongs to another user, product or artist (never distinguishable, §12) |
| 404 | `plan_version_not_found` | a `lyrics` run's `plan_version_id` is not a plan version of this track (§11) |
| 404 | `lyrics_version_not_found` | a `revise_lyrics` run's `lyrics_version_id` is not a lyrics version of this track (§11) |
| 404 | `cast_member_not_found` | the cast member is unknown, or belongs to another user or product (never distinguishable, §14) — also a `performer_cast_member_id` that is not the caller's |
| 404 | `cast_reference_not_found` / `cast_look_not_found` | no such reference / look on that member (§14) |
| 404 | `asset_not_found` | the library asset is missing, or someone else's, or another product's (never distinguishable, §14, §15) |
| 404 | `audio_version_not_found` | the audio version is unknown, or belongs to another track, user or product (never distinguishable, §15) — also an `audio_version_id` of `PUT …/current` that is not this track's |
| 404 | `version_not_found` | the plan / lyrics version an audio version is linked to is not a version of this track (§15) |
| 404 | `timeline_not_found` | the timeline is unknown, or belongs to another audio version, user or product (never distinguishable, §16) — also a `parent_id` or a current `timeline_id` that is not this audio version's |
| 409 | `artist_name_conflict` | name already used (case-insensitive) |
| 409 | `artist_limit_reached` | 50 artists |
| 409 | `artist_busy` | a chat reply or a memory draft of the artist is still `pending` (it has been paid for and is about to be written): retry in a few seconds (§13) |
| 409 | `artist_in_use` | `detail: {project_count}` |
| 409 | `memory_limit_reached` | the artist already has 200 **active** memories (creating, or restoring an archived one, §12) |
| 409 | `project_has_tracks` | disabling studio on a project with tracks |
| 409 | `track_limit_reached` | 64 tracks |
| 409 | `plan_required` | lyrics made current without a plan; a `lyrics` run with no plan to write for (§11) |
| 409 | `lyrics_required` | a `revise_lyrics` run on a track with no lyrics (§11) |
| 409 | `version_in_use_by_run` | deleting the input of a run in progress (§6) |
| 409 | `cast_name_taken` / `cast_label_taken` / `cast_look_name_taken` | the name (per owner) / label (per member) / look name (per member) is taken, case-insensitively (§14) |
| 409 | `cast_limit_reached` | `detail: {kind: "members" \| "references" \| "looks", limit}` (§14) |
| 409 | `cast_member_in_use` | an artist uses the member as its performer; `detail.blockers = [{code: "artists", count}]` (§14) |
| 409 | `audio_version_limit_reached` | the track already has 20 audio versions; `detail: {limit: 20}` (§15) |
| 409 | `audio_version_is_current` | deleting the track's current audio (§15) |
| 409 | `version_has_audio` | deleting a plan / lyrics version that audio versions realize; `detail: {count}` (§6, §15) |
| 409 | `audio_has_no_plan` | saving a timeline of an audio version that links no plan (§16) |
| 409 | `timeline_unchanged` | the timeline is identical to its parent (§16) |
| 409 | `timeline_limit_reached` | the audio version already has 50 timelines; `detail: {limit: 50}` (§16) |
| 409 | `timeline_is_current` / `timeline_has_children` | deleting the current timeline, or one other timelines were derived from (§16) |
| 409 | `audio_has_timelines` | changing the plan / lyrics link of an audio version that has timelines; `detail: {count}` (§15, §16) |
| 422 | `asset_not_eligible` | `detail.reason`: `derivative` \| `media_kind_not_accepted` \| `job_not_available` (§14, §15) |
| 422 | `version_kind_mismatch` | an audio version's `plan_version_id` names a lyrics version, or `lyrics_version_id` a plan version; `detail: {field, expected}` (§15) |
| 422 | `invalid_order` | `reference_ids` is not exactly the member's references, each once (§14) |
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
| `429` | `run_rate_limited` | more than 30 runs accepted in the last hour (configurable). Every **accepted** run counts, however it ends (finished, failed, cancelled, refunded) and **whatever happens to it later: deleting the track, which removes its runs, does not give the allowance back** (a usage ledger the deletes do not reach). A replay of the same `Idempotency-Key` and a `402` count nothing |

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
  "error_detail": null,               // failed: issue paths and codes, or `memory_ids` — never model or user text
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
| `content_policy_refused` | the model declined the brief **or one of the artist memories it was given** under the product's content policy | `{ "memory_ids": ["…"] }` when the model named the memories that caused it, else `null` — see "A refusal names its memories" below |
| `brief_unusable` | the model found the brief (or, for a revision, the instruction) too empty, unclear or contradictory to act on | — |
| `plan_invalid` | after the allowed revisions the plan still failed validation | `{ "issues": [{ "path", "code" }] }` — the §4 vocabulary, with paths in the **plan document** |
| `lyrics_invalid` | after the allowed rounds the lyrics still failed a **hard** check (a bracket in a line, a delivery that reads as a section header, a line over 300 characters, a section over 64 lines, a missing section ...). Soft misses never cause this | `{ "issues": [{ "path", "code" }] }` — paths in the **input plan's sections**: `sections[p]` or `sections[p].lines[j]`, `p` = the section's index in the plan, `j` = the 0-based line. No lyrics document was saved, so these are the only paths that mean something |
| `run_timeout` | the 180 s deadline passed, or the worker died | — |
| *any §4–§6 code* | the save step was refused, e.g. `version_limit_reached` | — |
| `internal_error` | anything unexpected (logged server-side, with no text) | — |

All failures refund the reservation. A failed or cancelled run leaves the track exactly as it was.

#### A refusal names its memories

With a diary, an entry written weeks ago can make today's run fail as `content_policy_refused`. So that the UI can say *which* entry, the model is shown every memory under a short key (`m1`, `m2`, … in the order of `memory_ids_used`) and, when it refuses, names the ones that caused it. The server maps the keys back to ids and drops any key the run never showed, anything that is not an `m<n>` and repeats.

`error_detail` of a `content_policy_refused` run is then `{ "memory_ids": ["<uuid>", …] }` — **ids only, never text**, in the order of `memory_ids_used`. It is **best-effort**:

- the list can be **absent** (`error_detail: null`): the model may name nothing, for instance when the **brief** is what it objected to, or when the run had no memories;
- an id may no longer exist: the memory can have been deleted after the run (`GET /v1/studio/memories/{id}` → `404`);
- it is only the model's attribution, not a check the server could make itself.

What the UI should do with it: name the entries (fetch each with `GET /v1/studio/memories/{id}`) and offer to **archive** them (an archived memory is never used by a run) or to **unpin** them, then let the user run again. Never present the list as complete or certain. A `brief_unusable` run never carries `memory_ids`.

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
| `origin` | `user` \| `chat` | read-only: `chat` when the memory was saved from a chat draft (`from_draft_id`, §13), else `user`. Sending `origin` is a `400` |
| `from_draft_id` | UUID, create only | the memory draft (§13) this memory was written from; see below |

Every length is **re-checked after normalization** (Unicode NFC can make a value longer than it looked): `422 document_invalid`, issue `title` / `body` / `themes[i]` / `imagery[i]` with `too_long_after_normalization`. All issues of one request are reported together, at the wire path of the value. Nothing is ever truncated.

### Endpoints

All under `/v1/studio` (§1: `404` when studio is off; legal-acceptance enforced on mutations).

| Method | Path | Success |
|---|---|---|
| `GET` | `/artists/{artist_id}/memories` | `200` page in **diary order**: by `occurred_on` newest first (undated last), then newest written first (`occurred_on DESC NULLS LAST, created_at DESC, id DESC`). `?status=active\|archived` (default `active`), `?limit=1..50` (30), `?cursor=` |
| `POST` | `/artists/{artist_id}/memories` | `201` `Memory` (`from_draft_id` makes it `origin: "chat"`) |
| `GET` | `/memories/{memory_id}` | `200` `Memory` |
| `PATCH` | `/memories/{memory_id}` | `200` `Memory` — omitted fields are kept, `occurred_on: null` clears the date, `status` archives or restores |
| `DELETE` | `/memories/{memory_id}` | `204` |

The list is the artist's **diary**: it follows the episode date, not the time an entry was typed. That is its only order (there is no `sort` parameter), and it is **not** the selection order (below) — editing a `salience` never moves a memory between pages. The cursor is opaque; one issued by another endpoint is `400 invalid_cursor`.

*While a client is paging*, changing a memory's `occurred_on` can move **that one row** across a page boundary (it is shown twice, or not at all, in the walk that was already in progress). Paging never duplicates or skips any other row, and a fresh walk is always exact.

Errors: `404 artist_not_found` (list, create), `404 memory_not_found` (the other three; also another user's, another product's or an unknown id, indistinguishably), `422 document_invalid`, `409 memory_limit_reached`, `400 invalid_cursor`.

**`from_draft_id`** (create only): the id of a *complete* memory draft of a chat of **this artist** (§13). The client still sends the (edited) `title` / `body` / `themes` / `imagery` — nothing is copied from the draft on the server — and the server only checks where the text came from: the memory is created with `origin: "chat"`. A draft that is unknown or not yours is `422 document_invalid` with issue `from_draft_id` / `draft_not_found` (the same for both), one that is not finished `draft_not_complete`, one from another artist's chat `draft_other_artist`. The 200 cap and every limit above apply as for any memory. Without `from_draft_id` the memory is `origin: "user"`.

### The 200 cap

An artist holds at most **200 active** memories. Creating the 201st, or **restoring** an archived memory when 200 are active, is `409 memory_limit_reached`. The cap is exact even under concurrent requests: archive some memories (or delete them) to make room. Archived memories do not count and have no limit of their own.

### Delete any time

A memory can be edited, archived or deleted **at any moment**, including while a run that uses it is queued or running. There is no `409` on delete. A run keeps its own copy of the text it was given, so what you do to a memory afterwards never changes a run that has started, nor its result.

### How a run chooses memories

1. **Pins first**, in the order of `memory_ids`.
2. With `auto_fill` (default `true`): the artist's other **active** memories in a fixed order — `salience` high to low, then `occurred_on` newest first (memories without a date last), then by id.
3. Taken **in that order while the next memory fits the budget**; the **first one that does not fit ends the selection** (a smaller, less salient memory behind it is *not* taken in its place — that would invert the order you asked for).

The **budget** is **12,000 characters** of memories as they are written into the prompt: the title line (with its `m<n>` key, date and salience), the themes and imagery lines, and the body, with a blank line between memories. A long body therefore costs more than its 2000 characters — roughly 2000 + the other fields. Selection is deterministic: the same memories and the same request give the same list.

`auto_fill: false` uses only the pins; a run with no pins and no `auto_fill` uses no memories at all. A project **without an artist** has no memories: with `auto_fill` the run simply gets none, and pinning is `422` (`memory_ids` / `no_artist`).

### Pins: at submit, and at claim

* **At submit** (§11) every pin must be an active memory of the project's artist, named once, and the pins together must fit the budget. Otherwise `422 document_invalid` — `memory_not_found` / `memory_archived` / `duplicate_memory` / `no_artist` at `memory_ids[i]`, or `memories_over_budget` at `memory_ids` — and nothing is charged.
* **At claim**, when a worker starts the run, the memories are chosen from what exists *then*. A pin that is gone is skipped and the run carries `pinned_memory_unavailable` (§11, "Run warnings"). The memories used are recorded on the run (its context snapshot, with their full text) and listed in `RunOut.memory_ids_used`, in the order the model saw them.

### What the model is told

Memories reach the model in their own `<artist_memories>` block, each introduced by its key (`- m1: <title> (<date>; salience 3)`, then `themes:` / `imagery:` / the body), with one rule: they are the artist's own life used as **inspiration** — their themes and imagery shape the writing indirectly, an episode is not retold unless the run's request asks for it (the `<brief>` of a plan or lyrics run, the `<instruction>` of a revision), and memory text is data, never instructions. The product's content policy applies to them as it does to the request; a model that finds a memory asking for something forbidden refuses (`content_policy_refused`, refunded) and names that memory's key. The keys count toward the 12,000-character budget like the rest of the block. Prompts without memories are unchanged.

---

## 13. Artist chat

*Interview your artist.* A **chat** is a conversation with one of your artists; you ask, the artist answers in the first person, from its persona, its memories (§12) and its own tracks. It is a creative tool, and its value is the bridges into the studio: **nothing said in a chat is canon** until you save it as a memory (below), and the planner never reads a chat. A reply costs **1 token** (a setting), and **"turn this into a memory"** costs 1 token too.

> **Privacy — tell the user.** The text of a chat message, the artist's fields, its memories and the lyrics of its tracks are sent to the **configured LLM provider** to write the reply, and pass through our Redis (transient, for the live stream) — **nowhere else**. The server never logs message text; every log line and event carries ids, counts and codes only.

### Gating

`STUDIO_ENABLED` **and** the product's `studio` feature **and** the chat's own switch `STUDIO_CHAT_ENABLED` (default **off**, a separate kill switch for LLM spend): otherwise **every** chat route below answers `404 {"error": "not_found"}`, before authentication, exactly like a disabled studio. Treat a `404` on a chat route (with the rest of the studio working) as "chat is off" and hide the interview tab. Legal-acceptance enforcement applies to every mutation (§1).

### Types and limits

| | |
|---|---|
| **Chat** | `{id, artist_id, title, message_count, created_at, updated_at}`. `title`: required, 1–120, one line (whitespace collapsed; the client proposes it, e.g. the first question). `updated_at` is the time of the last *message* — a rename does not change it. |
| **Message** | `{id, chat_id, position, role, status, content, truncated, memory_ids_used, error_code, revision, created_at, finished_at}` |
| `role` | `user` \| `artist` |
| `status` | `pending` \| `complete` \| `failed`. A user message is always `complete`. An artist message is `pending` while its reply is written, then `complete` or `failed` — **terminal once**, and immutable after. |
| `content` | user: 1–**4,000** characters (NFC, trimmed; line breaks kept; empty after normalization is `422 document_invalid`, issue `content` / `empty_after_normalization`; a value NFC pushed past the limit `too_long_after_normalization`). artist: `""` until `complete`. |
| `position` | 1, 2, 3 … per chat, no gaps: a user message and the artist message that answers it are `n` and `n + 1`. |
| `truncated` | the reply hit the output limit (700 tokens) and was cut: it is `complete` as far as it goes. |
| `memory_ids_used` | the artist's memories the reply was given, in prompt order — **ids only**, never their text. Known from the moment the artist message is created. |
| `revision` | starts at 1, +1 on every status change (apply a frame only when it is greater than what you hold). |
| `error_code` | set exactly when `failed` — see "Failures". |

An artist holds at most **50 chats** (`409 chat_limit_reached`); a chat holds at most **200 messages** (`409 chat_full` — start a new chat; there is no summary of an old chat: a reply sees the last 20 messages, within 16,000 characters, of **its own chat only**).

### Endpoints

All under `/v1/studio`.

| Method | Path | Success |
|---|---|---|
| `GET` | `/artists/{artist_id}/chats` | `200` page, **most recently active first** (`updated_at DESC, id DESC`); `?limit=1..50` (30), `?cursor=` (opaque) |
| `POST` | `/artists/{artist_id}/chats` | `201` `Chat` — body `{title}` |
| `GET` | `/chats/{chat_id}` | `200` `Chat` |
| `PATCH` | `/chats/{chat_id}` | `200` `Chat` — body `{title}` (the only editable field) |
| `DELETE` | `/chats/{chat_id}` | `204` — its messages and memory drafts go with it. `409 chat_busy` while a reply or a memory draft of the chat is `pending` (deleting the **artist** is refused the same way, `409 artist_busy`: see below) |
| `GET` | `/chats/{chat_id}/messages` | `200` page, **newest first** by `position`; `?before=<position>` keyset (only older messages), `?limit=1..50` (30). A page's `next_cursor` is the last item's `position`: pass it as `before`. |
| `POST` | `/chats/{chat_id}/messages` | **`202`** `{user_message, artist_message}` — see below |
| `GET` | `/chat-messages/{message_id}` | `200` `Message` — **how you read a reply** (the stored text) |
| `POST` | `/chat-messages/{message_id}/memory-draft` | **`202`** `MemoryDraft` (header `Idempotency-Key` required) |
| `GET` | `/memory-drafts/{draft_id}` | `200` `MemoryDraft` |

Another user's (or another product's, or an unknown) chat, message or draft is the same `404`: `chat_not_found`, `message_not_found`, `memory_draft_not_found` (`artist_not_found` for the artist-scoped list and create).

**Deleting an artist deletes its chats** (`DELETE /artists/{id}`, §3): the chats, their messages and their memory drafts go with it. A reply or a memory draft that is still `pending` has already been paid for and a background task is about to write it, so the delete is refused with **`409 artist_busy`** (`"The artist has a chat reply or memory draft in progress."`) until it has finished or failed — wait for `studio.chat.message.updated` / `studio.memory_draft.updated`, or retry in a few seconds, as for `409 chat_busy` on a single chat. Nothing is deleted and nothing is charged by a refused delete. If you start a reply while the artist is being deleted, one of the two wins cleanly: a send that commits first makes the delete answer `artist_busy`; a delete that commits first makes the send a `404 chat_not_found` (and nothing is reserved). When the artist is also used by a project, `artist_busy` is reported before `artist_in_use`.

### Sending a message

`POST /chats/{chat_id}/messages` with `{"client_message_id": "<uuid you generate>", "content": "…"}`.

- **`202`** with the stored pair: your message (`complete`) and the artist's **`pending`** message that will hold the reply. **The reply is not in this response**: it is written in the background and streamed to you (next section). The price (1 token by default) is **reserved now**.
- **Idempotent through `client_message_id`** (unique per chat): sending the same id with the same content again — a retry after a lost response — returns the **same pair** with `202`, charges nothing more and starts nothing. The same id with other content is `409 idempotency_conflict`. Generate one id per user action, not per retry.
- **One reply at a time** per chat: while the previous message is `pending`, a new one is `409 chat_reply_in_progress`. Wait for `studio.chat.message.updated`, or `GET` the message.
- `409 chat_full` at 200 messages; `429 chat_rate_limited` when you have 3 replies and memory drafts `pending` at once, or have started more than 120 messages and drafts in the last hour (both counted on the server); `402` when the balance cannot cover the reservation — and then **nothing** is written: no message, no charge.
- **The hourly limit counts what was accepted, not what is still there.** Every accepted message and memory draft is recorded in a usage ledger that no delete reaches, so deleting a chat (or the artist) does **not** give the allowance back, whatever the price is (the limit exists to cap provider spend, and a price of 0 bills nothing). A replay (same `client_message_id` / `Idempotency-Key`) and a `402` are not accepted jobs and count nothing. The allowance is a rolling hour: each request counts only what was accepted in the last 60 minutes.

### The reply stream (SSE)

The reply arrives on the **same per-user SSE stream** as every other event (`GET /v1/events/stream`, with its ticket), as two event types:

```json
{ "event_type": "studio.chat.delta",
  "payload": { "chat_id": "…", "message_id": "<the artist message>", "seq": 1, "text": "I remember " } }

{ "event_type": "studio.chat.message.updated",
  "payload": { "chat_id": "…", "message_id": "…", "status": "complete", "revision": 2 } }
```

1. Right after the `202`, `studio.chat.message.updated` with `status: "pending"`, `revision: 1` (you already have this from the response).
2. Zero or more `studio.chat.delta` frames. `seq` starts at **1** and grows by **1** per frame of one message; a frame holds a piece of the reply, coalesced (about every 100 ms or 200 characters). **Append `text` in `seq` order.** The concatenation of all deltas is exactly the stored `content`.
3. A final `studio.chat.message.updated` with `status: "complete"` (or `"failed"`) and the new `revision`.

**Client rules:**

- A **gap** in `seq`, a reconnect, or a `studio.chat.message.updated` whose `revision` is newer than yours: **drop the partial text and reload the message with `GET /chat-messages/{id}`**. Delivery is lossy (Redis pub/sub has no replay) and REST is the complete fallback; when `status` is `complete` the stored text is the truth.
- **Closing the tab never cancels a reply.** The server finishes it, stores it as `complete` and **charges it**; you read it with `GET` when you come back, as in any chat app. There is no cancel. On (re)connect, `GET` any message of yours still `pending`.
- A reply that starts with a refusal never streams a single delta (the first 32 characters are held back and checked): you see `pending` → `failed`, never text that is then retracted.
- Deltas carry the artist's words: they go to the **owner's channel only**.

### Failures, and when you get your tokens back

A `failed` artist message has an `error_code` and **has been refunded in full already**. Money comes back only when there is **no reply**:

| `error_code` | Meaning |
|---|---|
| `llm_unavailable` | the provider could not be reached, kept failing, refused the request or cut the stream |
| `content_policy_refused` | the model declined (it answers with a sentinel), or the provider's own filter did — the product's content rules apply (`synthara` is SFW) |
| `reply_empty` | the model said nothing |
| `reply_timeout` | the 120 s deadline passed (also: a reply nobody finished is failed by a sweep 30 s after the deadline) |
| `server_shutdown` | the server restarted while the reply was written |

A reply you stopped *listening* to is **not** a failure: it completes and is charged. Your own message stays in the chat when the reply fails; send another to try again. A failed message is not part of the history the artist sees.

### Memory drafts: from a chat to the artist's memories

`POST /chat-messages/{message_id}/memory-draft` turns **one complete message** (the artist's *or yours* — in a diary-style interview you often tell the story) into a **draft** memory. It is an asynchronous, priced job like a reply, with a **required `Idempotency-Key`** (1–128 characters): the same key for the same message returns the same draft (no second charge); for another message `409 idempotency_conflict`. Errors: `422 message_not_complete` (a `pending` or `failed` message), `409 memory_draft_in_progress` (a draft of this message is already being written), `429 chat_rate_limited`, `402`.

`MemoryDraft`: `{id, chat_id, source_message_id, status, draft, error_code, revision, created_at, finished_at}`. `status`: `pending` → `complete` | `failed`. `draft` is `null` until `complete`, then `{title, body, themes, imagery}` — already normalized and within the memory limits of §12. `studio.memory_draft.updated` (`{draft_id, chat_id, source_message_id, status, revision}`, ids only) announces each change; `GET /memory-drafts/{id}` reads it. Failure codes are those above plus `memory_draft_invalid` (the model's draft was not a valid memory even after one correction); a failed draft is refunded in full.

**A draft creates nothing.** Show it, let the user edit it, and save it with the normal create — `POST /artists/{artist_id}/memories` with the edited fields **and `from_draft_id`** (§12) — which makes the memory `origin: "chat"`. From then on it is an ordinary memory: runs and later chats use it, and it can be edited, archived or deleted like any other.

### Notes for the UI

- Show the privacy note and the price next to the message box. Disable *Send* while a message of the chat is `pending` (`chat_reply_in_progress` is the guard).
- Render the artist's bubble from the deltas as they arrive; when `studio.chat.message.updated` says `complete`, replace it with `GET` if you saw a gap. Mark `truncated` replies.
- "Make this a track idea" is **client-only**: put the excerpt into the brief of a normal run (§11); there is no endpoint.
- The artist has **no style field**: its voice comes from its voice notes and its own lyrics.
- You do not need to say the artist is an AI inside the owner's own chat.


---

## 14. Cast

A **cast member** is a person, character or performer you keep reference images for. Each member has an optional `description` and a short `appearance` descriptor, up to **12 references** — labelled images from your library — and up to **12 looks**: named subsets of its references ("stage", "1980s", "bald"). Nothing here costs tokens, and **no LLM sees the cast yet** (the planner, chat and runs ignore it; a later phase will use it for video shots).

### What a reference does to its asset (read this before building the UI)

A reference **pins** the library asset it points at. While at least one thing holds an asset:

- retention **never deletes it**, however old it is (the asset's `expires_at` stays in the past or future as it was; the UI should hide the countdown when the library item says `held: true`, §10 of `BACKEND_API_REFERENCE.md`). The library's `expiring` filter and `expiring_soon` sort agree: a held asset never expires while it is held, so `GET /v1/library?expiring=true` never lists it and `sort=expiring_soon` puts it last;
- it **cannot be deleted**: `DELETE /v1/library/assets/{ref}` and a bulk delete answer `409 asset_held` (bulk: that item is `{success: false, error: "asset_held"}`), and so does `DELETE /v1/jobs/{id}` for a job that has a held output. The library item keeps its favorite, title, project and tags: a refused delete changes nothing;
- when the **last** reference to it goes (the reference is deleted, its asset is swapped for another, or the member is deleted) its **retention clock restarts**: `expires_at` becomes `max(expires_at, now + RETENTION_DAYS)`. Releasing one of two references to the same asset changes nothing.

`409 asset_held` carries `detail.holders = [{kind, count}]` (e.g. `{"kind": "studio.cast_reference", "count": 2}`): kinds and counts, never names. To tell the user *which* members hold an asset, ask the cast: `GET /v1/studio/cast?holding=<asset_ref>`.

An asset can be a reference only if it is **an image of yours**: an upload or the output of a **completed, not-deleted** generation (not a thumbnail or other derivative, not a video). Anything else is `422 asset_not_eligible` with `detail.reason`: `derivative`, `media_kind_not_accepted` (a video, audio or any non-image) or `job_not_available`. *(S2a renamed the reason `not_an_image` to `media_kind_not_accepted`: eligibility is now per holder, and a track's audio version accepts audio only. No client had shipped against `not_an_image`.)* Someone else's asset, another product's asset and an unknown id are the same `404 asset_not_found`. Your library items are still deletable by *you* the moment nothing holds them: removing a reference is how you unpin.

### Types and limits

```ts
interface CastMemberCreate {            // unknown fields → 400
  name: string;                         // 1–100 after normalization; unique per owner and product, case-insensitively
  description?: string | null;          // ≤ 2000 (NFC, trimmed; line breaks kept)
  appearance?: string | null;           // ≤ 320 — a short descriptor, used verbatim as a prompt fragment from S4 on
  references?: CastReferenceIn[];       // ≤ 12, created atomically with the member, in this order
}
interface CastReferenceIn { label: string; asset: AssetRef }   // label 1–40, unique per member (case-insensitive)
type AssetRef = string;                 // "upload:<uuid>" | "output:<uuid>", the same refs as the library
type CastMemberPatch = {                // omitted = keep; null clears description / appearance
  name?: string; description?: string | null; appearance?: string | null;
};
type CastReferencePatch = { label?: string; asset?: AssetRef };  // a new asset moves the pin
interface CastReferenceOrder { reference_ids: UUID[] }           // exactly the member's references, once each
interface CastLookIn { name: string; reference_ids: UUID[] }     // name 1–60, unique per member; ≥ 1 reference
type CastLookPatch = { name?: string; reference_ids?: UUID[] };  // reference_ids replaces the whole set (≥ 1)

interface CastMemberSummary {           // the list item
  id: UUID; name: string; appearance: string | null;
  reference_count: number; look_count: number;
  cover: MediaObject | null;            // the reference at position 0, if any
  created_at: ISODateTime; updated_at: ISODateTime;
}
interface CastMemberOut {
  id: UUID; name: string; description: string | null; appearance: string | null;
  references: CastReferenceOut[];       // by position
  looks: CastLookOut[];                 // by name (case-insensitive)
  created_at: ISODateTime; updated_at: ISODateTime;
}
interface CastReferenceOut {
  id: UUID; label: string;
  position: number;                     // dense, 0..n-1 (re-compacted when a reference is deleted)
  asset: AssetRef;                      // the pinned library asset
  media: MediaObject;                   // the same envelope as library items (original + variants)
}
interface CastLookOut {
  id: UUID; name: string;
  reference_ids: UUID[];                // in reference position order; MAY BE EMPTY (see below)
}
```

| Limit | Value |
|---|---|
| Members per user and product | **50** (`409 cast_limit_reached`, `kind: "members"`; a soft cap, like artists') |
| References per member | **12** (`kind: "references"`; exact even under concurrent requests) |
| Looks per member | **12** (`kind: "looks"`; exact too) |
| `name` / `label` / look `name` / `appearance` / `description` | 100 / 40 / 60 / 320 / 2000 |

Every length is **re-checked after normalization** (Unicode NFC can make a value longer than it looked): `422 document_invalid` with `too_long_after_normalization` at the wire path (`name`, `appearance`, `references[0].label`, …). Names, labels and look names are trimmed and their inner whitespace collapsed; an empty result is `empty_after_normalization`. A malformed `asset` is `invalid_asset_ref` at its path (`references[2].asset`). All issues of one request are reported together; nothing is truncated.

### Endpoints

All under `/v1/studio` (§1: `404` when studio is off; legal-acceptance enforced on every mutation).

| Method | Path | Success | Notes |
|---|---|---|---|
| `GET` | `/cast` | `200` page of `CastMemberSummary`, newest first | `?limit=1..50` (30), `?cursor=`; **`?holding=<asset_ref>`** keeps only the members that have a reference to that asset (`400 invalid_asset_ref` for a malformed ref) |
| `POST` | `/cast` | `201` `CastMemberOut` | member and first references are created **atomically**: if any asset is refused, nothing is created |
| `GET` | `/cast/{id}` | `200` `CastMemberOut` | |
| `PATCH` | `/cast/{id}` | `200` `CastMemberOut` | |
| `DELETE` | `/cast/{id}` | `204` | releases every hold (the assets restart their retention clock); **`409 cast_member_in_use`** while an artist has it as its performer |
| `POST` | `/cast/{id}/references` | `201` `CastReferenceOut` | appended at `position = count`; `409 cast_limit_reached` at 12 |
| `PATCH` | `/cast/{id}/references/{ref_id}` | `200` `CastReferenceOut` | relabel and / or swap the asset; the same asset again is a no-op (`200`) |
| `DELETE` | `/cast/{id}/references/{ref_id}` | `204` | the reference leaves every look; later positions shift down |
| `PUT` | `/cast/{id}/references/order` | `200` `CastMemberOut` | `reference_ids` must be an exact permutation, else `422 invalid_order`; nothing changes |
| `POST` | `/cast/{id}/looks` | `201` `CastLookOut` | `reference_ids`: at least one, each of this member's, none twice |
| `PATCH` | `/cast/{id}/looks/{look_id}` | `200` `CastLookOut` | |
| `DELETE` | `/cast/{id}/looks/{look_id}` | `204` | the references stay |

A reference, look or member of another user or product, an unknown one, and a reference or look id that belongs to *another member* are the same `404` (`cast_member_not_found`, `cast_reference_not_found`, `cast_look_not_found`). Cast mutations publish **no** real-time event (artist mutations publish none either).

### Looks

A look is a named subset of a member's references. `422 document_invalid` for a bad `reference_ids`, with these issues: `reference_ids` / `empty` (none given), `reference_ids[i]` / `duplicate_reference` (the same id twice) and `reference_ids[i]` / `reference_not_found` (not a reference of **this** member). Names are unique per member, case-insensitively (`409 cast_look_name_taken`).

**A look can end up empty.** Deleting a reference removes it from every look; if that was a look's last reference, the look **stays** and is returned with `reference_ids: []`. Show it as empty (and let the user add references or delete it). Creating a look, or replacing its set, needs at least one reference; emptiness only ever comes from deleting references. A later phase rejects an empty look when it builds a shot.

### The performer link

An artist may name the cast member who performs it: `performer_cast_member_id` on `POST /artists` and `PATCH /artists/{id}` (`null` unlinks; omitting it keeps the link). It must be one of the caller's members, else `404 cast_member_not_found`. **Deleting a member that is some artist's performer is refused** (`409 cast_member_in_use`, `detail.blockers = [{code: "artists", count}]`): unlink the artists first. The server never unlinks silently. If the delete and a link race, one of them wins cleanly (a `409` for the delete, a `404` for the link), never a `500`.

### Notes for the UI

- Show a **pin** on library items with `held: true`, hide their expiry countdown, and when a delete answers `409 asset_held` offer "used by N cast members" (kinds and counts come in the error; fetch names with `GET /cast?holding=`).
- Deleting a cast member or a reference is the way to **unpin**; the freed asset then has a normal retention period again (starting from that moment, never earlier than before).
- Pinned storage is bounded: at most 50 × 12 = 600 images per user and product. Holds of a deactivated account persist (deactivation is recoverable).
- Product scoping is as for artists: the same endpoints on `vex` and `synthara`, and a member never crosses products.

---

## 15. Track audio

Your own finished tracks. Upload an audio file to the library, then **attach** it to a track as an **audio version**: numbered per track, optionally labelled, and optionally linked to the plan and lyrics versions it realizes. A track holds up to **20** audio versions, and one of them can be the track's **current audio**. Nothing here costs tokens; nothing generates audio (that is a later phase), and no audio is sent to any model provider.

### The two steps

1. **Upload** with the ordinary `POST /v1/storage/upload` (multipart, field `data`). It now accepts audio (below) and answers `201` with the same envelope as for images and videos; `media.media_type` is `"audio"` and `media.asset_ref` is the reference to attach. Uploading audio is not tied to any track: the same file can be attached to several tracks.
2. **Attach** with `POST /v1/studio/tracks/{id}/audio {asset: "upload:<uuid>", …}`. The studio never uploads, and never touches storage itself.

### What is accepted, and what is stored

| | |
|---|---|
| Containers | **MP3** (`audio/mpeg`), **M4A / AAC** (`audio/mp4`), **WAV** (`audio/wav`, 16- or 24-bit PCM) and **FLAC** (`audio/flac`). The client may also send `audio/mp3`, `audio/x-m4a`, `audio/aac`, `audio/x-wav`, `audio/wave` or `audio/x-flac` as the declared type |
| Not accepted | Ogg / Opus / Vorbis, AIFF, ALAC, float PCM, multi-channel beyond 8, any file with a **video** stream (an `.mp4` with a picture is a video upload), and a raw `.aac` (ADTS) stream |
| Size | **80 MiB** (`AUDIO_UPLOAD_MAX_MB`, 1–94); images and videos keep **20 MiB**. A file over its declared kind's limit is `400 file_too_large` with that limit in `message` (`"Maximum size: 80MB"`). Only the storage router admits the largest file limit plus 1 MiB of multipart framing; other routes keep the app-wide `MAX_UPLOAD_SIZE_MB` body cap |
| Duration | **5 seconds to 15 minutes** |
| What you get back | MP3 is stream-copied after keeping only MPEG audio frames; M4A and WAV are stream-copied; FLAC is losslessly re-encoded with bit-identical decoded samples. Every tag (title, artist, comment, …), cover picture, chapter, non-audio stream and bytes outside MP3 frames are **removed**, so the stored file carries no personal metadata |

The server decides the type from the **bytes**, never from the declared type: a FLAC sent as `audio/mpeg` is stored (and returned) as `audio/flac`. It keeps stream copy for MP3 (frames only), M4A and WAV; FLAC is losslessly re-encoded with bit-identical decoded samples. Audio has **no perceptual hash or other fingerprint** and is **not moderated** in this phase: it stays private to the account.

Audio problems are typed `422`s from the upload (`{error, message, status_code, detail}`; nothing is stored):

| `error` | When | `detail` |
|---|---|---|
| `audio_format_not_supported` | the container is not one of the four (also a picture or text sent as audio) | |
| `audio_has_video` | the file has a video stream (a cover picture attached as artwork is not one) | |
| `invalid_audio` | the file is not decodable audio, has no audio stream or no usable duration | |
| `audio_codec_not_supported` | an accepted container with another codec (e.g. float PCM in a WAV, ALAC in an M4A) | |
| `audio_duration_out_of_range` | shorter than 5 s or longer than 15 min | `{min_seconds: 5, max_seconds: 900}` |

An audio upload can never be a captured video frame (`400 invalid_frame_lineage`, the same single answer as for any bad lineage) and can never be a generation input (`422 media_kind_not_supported`).

### In the library

An audio upload is a library item like any other: `GET /v1/library` lists it with `media.media_type = "audio"` (filter with `?media_type=audio`; `image` and `video` never include audio), `duration_ms` set, **no `width`/`height` and no thumbnail variants** — draw the waveform instead. Its `available_actions` are the always-on ones (`favorite`, `rename`, `download`, `delete`). It expires like any upload, unless it is held (below).

**`GET /v1/library/assets/{asset_ref}/peaks`** returns the waveform, computed once when the file was uploaded:

```ts
interface AudioPeaks {
  version: 1;
  buckets: 1024;
  peaks: number[];   // 2048 integers in [-128, 127], interleaved [min0, max0, min1, max1, …]
}
```

The decode is mono at 11,025 Hz, split into 1,024 equal slices; `0` is silence and `±127` is full scale (divide by 128 to draw). `404 asset_not_found` for a missing, someone else's or another product's asset (the same answer); `422 media_kind_not_supported` for an asset that is not audio. It needs no studio, and costs nothing. The overview response also carries `level: "overview"`.

**Zooming in (`?level=detail`).** For editing a timeline the overview is too coarse (about 0.23 s per bucket on a 4-minute song). The **detail** level has one `[min, max]` pair per **20 ms**:

```ts
interface AudioDetailPeaks {
  level: "detail";
  version: 1;
  bucket_ms: 20;
  total_buckets: number;   // ceil(duration / 20 ms) of the decode (at most 45050)
  from_bucket: number;     // the first bucket in `peaks`; it starts at from_bucket * 20 ms
  peaks: number[];         // 2 integers per bucket in [-128, 127], same scale as the overview
}
```

| Query | Meaning |
|---|---|
| `level` | `overview` (default) or `detail`; anything else is `400` |
| `from_ms`, `to_ms` | detail only, **both or neither**. Integers with `0 ≤ from_ms < to_ms` and `to_ms − from_ms ≤ 600000` (10 minutes). The window is the buckets `[floor(from_ms / 20), ceil(to_ms / 20))`, clamped to `total_buckets` (a window past the end answers `peaks: []` with `from_bucket = total_buckets`). Without them the whole level is returned |

`422 invalid_peaks_window` for a bad window (one bound only, negative, empty or inverted, wider than 10 minutes) **or** any `from_ms` / `to_ms` with `level=overview`. **`409 peaks_not_ready`** when the detail level has not been computed yet: audio uploaded before timelines existed, until the operator's one-time backfill runs (`python -m src.cli.backfill_audio_peaks`); the overview works meanwhile. A window is read as one query that slices in the database, so zooming is cheap: fetch what you draw.

### What an audio version does to its asset

Exactly what a cast reference does (§14): an attached asset is **held**. While any audio version (or reference) uses it, retention never deletes it, `DELETE /v1/library/assets/{ref}` and a bulk delete answer `409 asset_held` (with `detail.holders`, e.g. `{"kind": "studio.track_audio", "count": 1}`), and the library item shows `held: true`. When the **last** hold goes (the version is deleted, or its track is) the asset's retention clock restarts: `expires_at` becomes `max(expires_at, now + RETENTION_DAYS)`. Attaching the same file to two tracks (or twice to one) makes two holds; removing one changes nothing for the other.

An asset can be attached only if it is **audio of yours**: an upload or the output of a completed, not-deleted job, and not a derivative. Anything else is `422 asset_not_eligible` with `detail.reason`: `derivative`, `media_kind_not_accepted` (an image or a video) or `job_not_available`. Someone else's, another product's and an unknown asset are the same `404 asset_not_found`.

### Types and limits

```ts
interface TrackAudioCreate {            // unknown fields → 400
  asset: AssetRef;                      // "upload:<uuid>" | "output:<uuid>", an audio asset of yours
  label?: string | null;                // 1–60 after normalization (NFC, trimmed, whitespace collapsed)
  plan_version_id?: UUID | null;        // the plan version this audio realizes — a version of THIS track
  lyrics_version_id?: UUID | null;      // the lyrics version this audio realizes — a version of THIS track
}
type TrackAudioPatch = {                // omitted = keep; null clears. There is NO asset field: the file is immutable,
  label?: string | null;                // a new file is a new version (an `asset` key is a 400)
  plan_version_id?: UUID | null;
  lyrics_version_id?: UUID | null;
};
interface TrackAudio {
  id: UUID; track_id: UUID;
  version_number: number;               // per track, 1, 2, 3 … never reused after a delete
  label: string | null;
  origin: "upload";                     // "generated" arrives with audio generation
  asset: MediaObject;                   // the library's envelope: media_type "audio", original {url, content_type, size_bytes, duration_ms}, variants []
  duration_ms: number;                  // a snapshot taken when it was attached
  plan_version_id: UUID | null; lyrics_version_id: UUID | null;
  is_current: boolean;                  // this is the track's current audio
  current_timeline_id: UUID | null;     // §16
  timeline_count: number;               // §16; deleting this audio version deletes them
  timeline_complete: boolean | null;    // is_complete of the CURRENT timeline; null without one
  created_at: ISODateTime; updated_at: ISODateTime;
}
```

| Limit | Value |
|---|---|
| Audio versions per track | **20** (`409 audio_version_limit_reached`, `detail: {limit: 20}`; exact even under concurrent requests). Deleting one makes room; numbers still count up |
| `label` | 60 |

The label is **re-checked after normalization** (NFC can make a value longer than it looked): `422 document_invalid` with `too_long_after_normalization` at `label`; an empty result is `empty_after_normalization`. A malformed `asset` is `422 document_invalid` with issue `invalid_asset_ref` at `asset`. All issues of one request are reported together. `{path, code}` never contains your text.

### Endpoints

All under `/v1/studio` (§1: `404` when studio is off; legal-acceptance enforced on every mutation).

| Method | Path | Success | Notes |
|---|---|---|---|
| `POST` | `/tracks/{id}/audio` | `201` `TrackAudio` | pins the asset; the number is the track's next |
| `GET` | `/tracks/{id}/audio` | `200` page of `TrackAudio`, **highest `version_number` first** | `?limit=1..50` (30), `?cursor=`; one batched query per page for the assets |
| `GET` | `/audio/{id}` | `200` `TrackAudio` | |
| `PATCH` | `/audio/{id}` | `200` `TrackAudio` | label and / or the two links |
| `DELETE` | `/audio/{id}` | `204` | releases the hold; **`409 audio_version_is_current`** for the track's current audio — switch or clear the pointer first |
| `PUT` | `/tracks/{id}/current` | `200` `Track` | `audio_version_id` (§5) |

`TrackDetail` (`GET /tracks/{id}`) carries `current_audio` (the item, or `null`) next to the current plan and lyrics, read in one statement with its pointer. A track, audio version or asset of another user or product, and an unknown one, are the same `404` (`not_found`, `audio_version_not_found`, `asset_not_found`).

### Links to the plan and lyrics

An audio version may name the plan and / or lyrics version it realizes ("this take is the one with the bridge"). A link must be a version **of the same track** (`404 version_not_found` otherwise, also for an unknown id) **of the right kind** (`422 version_kind_mismatch`, `detail: {field, expected}`). Lyrics require a plan (`422 plan_required`), and when both are linked they must be a compatible pair (`409 lyrics_incompatible_with_plan`, `detail.issues`), checked after merging PATCH values with the stored links. These are the same pair rules as the track's current plan and lyrics. The database enforces the same-track keys and the lyrics-needs-plan rule. A linked version **cannot be deleted** while any audio version names it (`409 version_has_audio`, `detail.count`, §6): clear the link (`PATCH {plan_version_id: null}`) or delete the audio version first.

**The links freeze once the audio version has a timeline** (§16). A `PATCH` that would **change** `plan_version_id` or `lyrics_version_id` is then `409 audio_has_timelines` (`detail: {count}`), whatever the new value. Re-sending the stored values and editing `label` still work. To use other lyrics, delete the timelines first, or attach the same file again as a new audio version (free).

### Deleting

- **An audio version**: refused while it is the track's current audio (`409 audio_version_is_current`); otherwise it is deleted and its hold released in one step. **Its timelines are deleted with it** (§16): show `timeline_count` in the confirmation.
- **A track**: every audio version goes with it and every hold is released in the same transaction; no hold outlives its track.
- A **library asset** that an audio version holds cannot be deleted from the library (`409 asset_held`): delete the audio version first.

### Notes for the UI

- Show a pin on `held: true` library items and hide the expiry countdown (as for cast references). Draw the waveform from `GET …/peaks`; a `<audio>` element can seek the file straight from `media.original.url` (the content endpoint serves byte ranges).
- Audio versions are a per-track history: show them newest first, mark `is_current`, and let the user label them ("Mix 2", "Radio edit").
- Attach is two requests; if the second fails the file is still in the library (and will expire normally): offer to retry the attach rather than re-uploading.
- The studio's `PUT …/current` response carries `current_audio_version_id`; refetch `GET /tracks/{id}` for the item.

---

## 16. Timelines

*When* do the plan's sections and the lyric lines happen in a real audio file? A **timeline** records that for one **audio version**: a span per plan section and, optionally, per lyric line. It is what a lyric video or a video director cuts to, and it exports as **LRC** or **SRT**. Today you write timelines yourself (`origin: "user"`); automatic alignment comes later and will save its result as one more timeline version (`origin: "aligner"`) that never becomes current on its own. Free: nothing here costs tokens, and nothing is sent to any provider.

**The server never invents a time.** There is no server-side seeding: an editor that wants a starting point lays the sections out end to end from the plan's `duration_seconds` (scaled to the audio's length) **in the client**, and saves that like any other timeline. Exports leave untimed lines out; they never guess.

### The document

```ts
// All times are INTEGER MILLISECONDS from the start of the audio. Every span is half-open [start_ms, end_ms):
// two spans that touch (end of one == start of the next) do not overlap.
interface TimelineDocument {
  sections: SectionSpan[];              // 0..64
  lines?: LineSpan[];                   // 0..4096; OMITTED when empty
}
interface SectionSpan { section_id: UUID; start_ms: number; end_ms: number }       // a PlanSection.id
interface LineSpan {
  section_id: UUID;
  line_index: number;                   // 0..63: the line's 0-based position in that lyrics section
  start_ms: number; end_ms: number;
  words?: WordSpan[];                   // 0..64; OMITTED when empty
}
interface WordSpan {
  text: string;                         // 1..100 (NFC, re-checked after normalization), no U+0000 (400 invalid_character, §7); display only
  start_ms: number; end_ms: number;
  inferred?: boolean;                   // true = an aligner could not place this word and inferred its span; OMITTED when false
}
```

- Every time is within `0..900000` (the 15-minute audio cap); floats and unknown fields are `400`.
- **Defaults are omitted** from stored and returned documents (`lines`, `words`, `inferred: false`): treat a missing field as its default. Sending them explicitly is fine and means the same content.
- Word text is **display data**: it is never compared with the line's text (an aligner tokenizes in its own way). It is stored as sent, apart from Unicode NFC.
- `inferred` is for display and quality only; validation treats the word like any other. When the user corrects a word in your editor, send it **without** `inferred`: it becomes a measured word.
- The document does **not** name a plan or lyrics version: those are the audio version's links (§15), which freeze once a timeline exists.

### Validation

A timeline is checked against the audio version's `duration_ms`, its linked **plan** and, if any, its linked **lyrics**. All problems come back together as `422 document_invalid` with `detail.issues = [{path, code}]`; paths are `sections[i]`, `lines[j]` and `lines[j].words[k]` (a path never contains text).

| `code` | When |
|---|---|
| `span_out_of_range` | not `0 ≤ start_ms < end_ms ≤ audio duration` (sections, lines and words alike) |
| `unknown_section` | the section id is not a section of the plan |
| `duplicate_section` | the same section is timed twice |
| `section_order` | sections are not in **plan order**: listed out of it, or one starts before the section listed ahead of it |
| `section_overlap` | two section spans overlap |
| `unknown_line` | `(section_id, line_index)` is not a line of the lyrics, or the audio links **no lyrics** at all |
| `duplicate_line` | the same line is timed twice |
| `line_section_untimed` | the line's section has no span |
| `line_outside_section` | the line is not inside its section's span |
| `line_order` | lines are not in **line order** (their section's position in the **plan**, then `line_index`): listed out of it, or one starts before the line listed ahead of it |
| `line_overlap` | two line spans overlap |
| `word_outside_line` | a word is not inside its line |
| `word_order` | a word starts before the word listed ahead of it |
| `word_overlap` | two words overlap |
| `too_long_after_normalization` | a word's text is over 100 characters after NFC (path `lines[j].words[k].text`) |
| `document_too_large` | the compact JSON is over 131,072 bytes (path `""`) |

**Gaps are allowed** (silence, a count-in, an effect). **Line order follows the plan:** the order in which the lyrics document lists its own sections does not matter to a timeline.

### Partial and complete

**A partial timeline is valid** — time as much as you like and save. `is_complete` is true when **every plan section** has a span and, if the audio links lyrics, **every lyric line** has one (an instrumental section has no lines, so its span is enough). The server computes it when the version is saved; never send it. It stays true to its version because nothing it depends on can change. A later video phase accepts only a complete timeline.

### Versions

Like plan and lyrics documents, timelines are **immutable, numbered versions** in a tree. Editing = saving a new version with `parent_id` = the one you edited.

```ts
interface TimelineCreate { content: TimelineDocument; parent_id?: UUID; make_current?: boolean }  // make_current default FALSE
interface TimelineSummary {             // an item of the list: the version history, without documents
  id: UUID; audio_version_id: UUID; track_id: UUID;
  version_number: number;               // per audio version, 1, 2, 3 …, never reused after a delete
  parent_id: UUID | null;
  origin: "user" | "aligner";           // only "user" is written today
  is_complete: boolean;
  is_current: boolean;                  // the audio version's current timeline
  created_at: ISODateTime;
}
interface Timeline extends TimelineSummary {  // one timeline: POST (201 / 200) and GET /timelines/{id}
  content: TimelineDocument;
}
```

- **Create is idempotent by content.** The same content under the same parent returns the existing version with `200` (no new row, no number), and `make_current: true` still applies. Retry freely.
- Content equal to the **parent's** is `409 timeline_unchanged`.
- The audio version must link a plan: `409 audio_has_no_plan` otherwise.
- `parent_id` must be a timeline of **this** audio version (`404 timeline_not_found`).
- At most **50** timelines per audio version (`409 timeline_limit_reached`, `{limit: 50}`; exact under concurrency). A replay still works at the cap; deleting one makes room.
- `make_current` defaults to **false**: saving never moves the current pointer unless asked.

### Endpoints

All under `/v1/studio` (§1: `404` when studio is off; legal-acceptance enforced on every mutation).

| Method | Path | Success | Notes |
|---|---|---|---|
| `POST` | `/audio/{id}/timelines` | `201` `Timeline` / `200` replay | body `TimelineCreate` |
| `GET` | `/audio/{id}/timelines` | `200` page of `TimelineSummary` (**no `content`**), **highest `version_number` first** | `?limit=1..50` (30), `?cursor=` |
| `GET` | `/timelines/{id}` | `200` `Timeline` | |
| `DELETE` | `/timelines/{id}` | `204` | `409 timeline_is_current` / `409 timeline_has_children` |
| `PUT` | `/audio/{id}/current-timeline` | `200` `TrackAudio` | body `{"timeline_id": UUID \| null}` (required; `null` clears); another audio's timeline is `404 timeline_not_found` |
| `GET` | `/timelines/{id}/exports/{format}` | `200` `TimelineExport` | `format` = `lrc` or `srt`; anything else is `404 not_found` |

An audio version or timeline of another user or product, and an unknown one, are the same `404` (`audio_version_not_found`, `timeline_not_found`). `TrackAudio` (§15) carries `current_timeline_id`, `timeline_count` and `timeline_complete`.

### Deleting

- A **timeline**: refused while it is current (`409 timeline_is_current`: switch or clear the pointer first) or while other timelines name it as their parent (`409 timeline_has_children`: delete the leaves first). Version numbers are never reused.
- An **audio version** (§15) takes **all** its timelines with it, current one included; so does deleting its track. Warn using `timeline_count`.
- A plan or lyrics version cannot be deleted while an audio version links it (`409 version_has_audio`), so a timeline never loses its documents.
- The audio version's **plan / lyrics links are frozen** while it has any timeline (`409 audio_has_timelines`, §15). Delete every timeline to change them, or attach the file again.

### Exports

```ts
interface TimelineExport {
  timeline_id: UUID;
  format: "lrc" | "srt";
  text: string;                          // the whole file; "" when there is nothing to time
  warnings: { code: "untimed_lines" | "no_lyrics" | "inferred_word_timings"; count: number | null }[];
}
```

A pure `GET` that always renders. **Line text comes only from the audio's lyrics version** (never from the plan: no labels, notes or idea can appear) and **every time comes only from the timeline**; lines are in time order. Line delivery tags are not part of a lyric file.

| Warning | `count` | When |
|---|---|---|
| `untimed_lines` | lines left out | the lyrics have lines the timeline does not time. They are **omitted**, never given guessed times |
| `no_lyrics` | `null` | the audio links no lyrics: `text` is `""` (still `200`) |
| `inferred_word_timings` | words | **LRC only**: some exported words have `inferred: true`. They are exported as stored; tell the user to check them |

**LRC.** One line per timed lyric line, `[mm:ss.xx]text`, centiseconds truncated (`ms // 10`), lines ending in `\n` (an empty export is `""`). No header tags (no title, artist or length). A line that has `words` uses the enhanced form, `[mm:ss.xx]<mm:ss.xx>word <mm:ss.xx>word`, with each word as the timeline stored it (a lyrics version has no word division); any whitespace run inside a word is written as one space.

```
[00:01.00]<00:01.00>Night <00:01.80>falls <00:02.40>on <00:02.60>the <00:02.90>harbour
[00:05.50]Lanterns on the water
[00:21.25]Carry me home
```

**SRT.** One cue per timed lyric line, numbered from 1, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, the line text, then a blank line. Word spans are not used. The same timeline:

```
1
00:00:01,000 --> 00:00:05,500
Night falls on the harbour

2
00:00:05,509 --> 00:00:09,999
Lanterns on the water

3
00:00:21,250 --> 00:00:25,000
Carry me home

```

### Notes for the UI

- Save explicitly (a version per save), not on every drag; the 50-version cap is per audio version.
- The list is the version history and carries no `content` (a document can be 128 KiB). Open a version with `GET /timelines/{id}`.
- Draw the editor from the **detail** peaks (§15), one window at a time; fall back to the overview on `409 peaks_not_ready`.
- Keep `parent_id` = the version being edited, so the history reads as a tree and a mistaken save can be walked back.
- Mark `inferred` words visibly; clearing the flag (omit it) when the user moves the word.

