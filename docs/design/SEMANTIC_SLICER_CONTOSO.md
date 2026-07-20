# Semantic Slicer v2 — Contoso, multi-dimension, two backends, voice

**Design spec · draft for review · 2026-07-20**

Natural-language (typed or spoken) queries that deterministically filter a standard
Contoso sales semantic model across **every dimension at once** — built two ways, side
by side:

- **Backend A — pure Fabric**: dim-attribute vectors in **SQL database in Fabric**
  (`VECTOR` + `VECTOR_DISTANCE`), embeddings from Azure OpenAI / Fabric's prebuilt AI.
  No rlat anywhere in the path.
- **Backend B — rlat**: one row-mode `.rlat` per dimension in OneLake, served by the
  existing Fabric User Data Function (`slice_stream`), local ONNX encoder, no external
  AI service.

Both backends return the same contract — a per-dimension set of business keys with
scores and "why it matched" receipts — and one front end consumes them: a **Fabric App
(Rayfin, preview)** with a search box, a push-to-talk microphone, visuals that recompute
on the fly, and a panel showing exactly which dimension members are filtered and why.

---

## 1. Goals

1. **Generalise the v1 semantic slicer from one dimension to a whole star schema.**
   v1 (the Airbnb London write-up, `docs/site/semantic-slicer.html`) proved the loop on a
   single 96k-row dimension with free-text descriptions. v2 targets a standard Contoso
   sales model: for **each** dim table, encode a vector per **unique combination of
   attribute values**; every dim row maps to its combination's vector; one query filters
   Product, Customer and Store simultaneously.
2. **A Fabric-only build.** Vectors are GA in SQL database in Fabric — show that the
   slicer can be built with nothing but Fabric-native parts, for users who want that.
3. **A fair side-by-side.** Same data, same front end, switchable backend — so the demo
   doubles as an honest comparison (infra, latency, receipts, cost-at-idle,
   determinism).
4. **Voice input.** A user speaks; the utterance is transcribed in the browser; the text
   drives the same pipeline; visuals recompute live.
5. **Deterministic application.** No LLM decides what gets filtered. Retrieval is
   ranked vector math with fixed thresholds and explicit tie-breaks; the resulting key
   sets are applied as plain set predicates in DAX. Same query in → same filters out.

### Non-goals

- **Structured intent** ("under $150", "in March 2024"). Numeric and date constraints
  are explicitly out of scope for the vector path — v1's honest boundary stands: a
  normal column filter is the right tool there. Numeric/date attributes are excluded
  from combo text (age is banded into labels as the one exception).
- **NL→DAX generation** (Copilot, Fabric data agents). Positioned against, not built
  (§15).
- **Query decomposition by LLM** (splitting "silver audio for young professionals" into
  per-dim clauses). Optional future extension; it would break the no-LLM-in-the-loop
  property, so the core design matches the whole utterance against every dim and lets
  per-dim gating decide (§4.3).
- Production hardening (multi-tenant auth, CI/CD, RLS interplay). This is a
  demo/reference architecture with production notes where they're cheap.

---

## 2. Prior art in this repo (what v2 reuses)

| Primitive | Where | Reused for |
|---|---|---|
| Row-mode builds — `(business_key, text)` → one passage per row, `passage_id` pinned to key, unique keys enforced | `build.walker.RowSourceWalker`, `build_rlat(row_mode=True)` | Backend B per-dim KM builds |
| OOM-safe full-corpus slicer UDF returning `{keys, hits[{key, score, text}]}` | `fabric/udf/function_app.py` → `slice_stream` | Backend B query path |
| 768-d query embedding endpoint (same warm encoder) | UDF `embed` | Backend A2 variant + parity checks |
| `.rlat` discovery, mtime-keyed cache, encoder revision pinning | `resonance_lattice.fabric` runtime | Backend B freshness story |
| Bit-exact CPU encoding guarantees (D1–D3) | `tests/harness/encoder_determinism.py` | Determinism claims (§13) |
| TREATAS / Execute-DAX pattern | v1 write-up + Data App | §10, upgraded to multi-dim `IN` predicates |

New in v2: multi-KM orchestration (per-dim fan-out + gating), the SQL-vector backend as
a first-class peer, the Rayfin front end, and voice.

---

## 3. Dataset — the Contoso sales model

**Default: SQLBI Contoso Data Generator V2, 100k orders** — ready-made downloads in
CSV / Parquet / **Delta** (drops straight into a lakehouse) plus PBIX
([tool](https://www.sqlbi.com/tools/contoso-data-generator/),
[ready-to-use releases](https://github.com/sql-bi/Contoso-Data-Generator-V2-Data/releases)).
Star schema: fact **Sales**, dims **Product, Customer, Store, Date, CurrencyExchange**.

Semantically rich dims and candidate attributes (⚠ verify exact V2 column names against
the [SQLBI details page](https://docs.sqlbi.com/contoso-data-generator/) before build —
spike S0):

| Dim | Attributes in combo text | Left out (structured, not semantic) |
|---|---|---|
| Product | Product Name, Brand, Category, Subcategory, Color, Manufacturer | Unit price/cost, weight, product code |
| Customer | Occupation, Gender, Age → **banded label** ("young adult", "middle-aged"…), City, State, Country, Continent | Birthday, lat/long, company id |
| Store | Store name, City, State, Country, Status | Square metres (optionally banded: "small/large format"), open/close dates |
| Date | — not encoded (non-goal) | everything |
| CurrencyExchange | — not encoded | everything |

Notes:
- V2 has **no Promotion dim and no Customer Education** attribute. If the demo wants
  those (they slice well semantically — "back to school promotion", "highly educated
  customers"), use classic **ContosoRetailDW**-shaped data (DimPromotion, Education,
  StoreType, product Class/Style) or extend V2. Default: ship with V2 and revisit.
- The semantic model is the **standard** Contoso model — measures like Sales Amount,
  Margin, Order Count over the star. Nothing about the model changes for the slicer;
  that's the point (the v1 lesson: "the model never changed").
- Model hosting: either the SQLBI **import-mode** PBIX published to the workspace, or a
  **Direct Lake** model over the Delta tables in a lakehouse. Both work identically
  here, because filters are injected per DAX query (§10) — there is **no write-back and
  no dependency on mirroring or reframing** in the core design. Direct Lake on OneLake
  is the more "Fabric-native" demo choice; import is the lowest-friction one.

---

## 4. Core design — combination vectors, per-dim retrieval, gated deterministic filters

### 4.1 One vector per unique attribute combination

For each encoded dim, build the deduplicated set of **attribute combinations**:

```
combo_text  = template(dim attributes)          e.g.
  Product : "Product: Contoso 4GB MP3 Player. Brand Contoso, Audio / MP3 & MP4
             Players, color Silver, by Contoso, Ltd."
  Customer: "Customer profile: teacher, female, middle-aged adult, lives in
             Newcastle, New South Wales, Australia (Oceania)."
  Store   : "Store: Contoso Sydney CBD store in Sydney, New South Wales,
             Australia; operating."
combo_hash  = sha256(combo_text)                 (identity across rebuilds)
combo_id    = dense rank of combo_text (ORDER BY combo_text)   (stable, deterministic)
members     = the set of dim business keys sharing that combination
```

Why combos and not raw rows:

- **Products** are mostly unique per row → combos ≈ rows; no loss.
- **Customers** collapse hard: tens of thousands of rows share a few thousand
  (occupation, gender, age-band, city) combinations. Embedding combos instead of rows
  cuts embedding cost and vector-store size by an order of magnitude, and — more
  importantly — makes matching semantics honest: two identical profiles can't rank
  differently.
- Expansion combo → member keys is a **deterministic join**, done after retrieval.

Template rules: fixed per dim, attribute order fixed, labels not bare values
("color Silver", not "Silver") so short attribute values carry context; NULLs render as
omitted clauses; numeric attributes excluded or banded (§1 non-goals). Templates are
versioned — `template_version` is stored beside the vectors, and changing a template
forces a rebuild of that dim (checked at build, not trusted to memory).

### 4.2 One query, every dim

The whole utterance is embedded **once**, and matched against **each dim's combo space
independently**. No clause splitting: each dim's combo space only contains its own
vocabulary, so "silver audio products for young professionals in Sydney" scores high
against Product combos on "silver audio", against Customer combos on "young
professionals", against Store/Customer geography on "Sydney" — and the junk matches are
suppressed by gating:

### 4.3 Dim gating — when does a dim participate at all?

Per dim, retrieval returns the top-K combos with cosine similarity. Then, with
**fixed, per-dim configured** parameters (no LLM, no heuristics at query time):

```
best      = max similarity over the dim's top-K
if best < floor_dim:          → dim is NOT filtered (gate closed; report "no match")
else: keep combos with  sim ≥ max(abs_min_dim, best − band_dim)
      cap at max_combos_dim, then expand members, cap at max_keys_dim
```

- `floor_dim` — does the query speak to this dim at all? ("show me quiet stores" must
  not accidentally filter Customer.) Calibrated per dim per backend (embedding spaces
  differ), from a small labelled query set + telemetry (§16 phase 5).
- `band_dim` — relative band under the best hit, so a decisive top match narrows the
  set and a diffuse match keeps it broad.
- Deterministic ordering everywhere: `ORDER BY similarity DESC, combo_id ASC`.
- If `max_keys_dim` would be exceeded (a very broad match), the dim **degrades to
  unfiltered** with an explicit "too broad — not applied" receipt rather than silently
  truncating meaning. (A "top-N only" mode is a config alternative.)

An unfiltered dim is not an error — it's the normal case for dims the query doesn't
address. The front end shows gate state per dim (§11).

### 4.4 The response contract (shared by both backends)

```jsonc
{
  "query": "silver audio products for young professionals in sydney",
  "backend": "fabric-sql",          // or "rlat"
  "embedMs": 41, "searchMs": 9, "cold": false,
  "dims": [
    { "dim": "Product",  "gated": true,  "best": 0.71,
      "keyColumn": "'Product'[ProductKey]",
      "keys": [4638, 2113, 977],
      "members": [ { "key": 4638, "label": "Contoso 4GB MP3 Player Silver",
                     "score": 0.71,
                     "receipt": "Product: Contoso 4GB MP3 Player. Brand Contoso, Audio / MP3 & MP4 Players, color Silver…" } ] },
    { "dim": "Customer", "gated": true,  "best": 0.63, "keyColumn": "'Customer'[CustomerKey]",
      "keys": [ /* expanded from ~40 combos */ ], "members": [ /* top combos as receipts */ ] },
    { "dim": "Store",    "gated": false, "best": 0.34, "keys": [] }
  ]
}
```

`members` carries **combo-level** receipts (the verbalised text + score) — that's the
explainability surface; `keys` carries the **expanded member keys** ready for DAX.

---

## 5. Architecture overview

```mermaid
flowchart LR
    subgraph Browser["Fabric App (Rayfin) — SPA"]
        MIC[Push-to-talk mic] -->|Azure Speech SDK / Web Speech| TXT[Query text box]
        TXT --> ORCH
        PANEL[Filter panel:\nper-dim gate state,\nmembers, scores, receipts]
        VIZ[Native charts + KPI cards]
    end

    ORCH{{UDF: semantic_slice\n(backend switch)}}
    TXT -.->|Entra bearer| ORCH

    subgraph A["Backend A — pure Fabric"]
        EMB[Query embedding\nAOAI / prebuilt ada-002]
        SQLDB[(SQL DB in Fabric\nsemslice.combo_* tables\nVECTOR + VECTOR_DISTANCE)]
        EMB --> SQLDB
    end

    subgraph B["Backend B — rlat"]
        RLAT[(OneLake Files/rlat/\ncontoso-product.rlat\ncontoso-customer.rlat\ncontoso-store.rlat)]
        ENC[Local ONNX encoder\nin UDF worker] --> RLAT
    end

    ORCH --> A
    ORCH --> B
    ORCH -->|response contract §4.4| PANEL
    ORCH -->|keys| DAX[Execute DAX Queries API\nCALCULATETABLE + IN {keys}]
    DAX --> SM[(Contoso semantic model\nimport or Direct Lake)]
    SM --> VIZ

    subgraph Build["Build pipeline (notebook, scheduled)"]
        DIMS[(Lakehouse:\nContoso Delta dims)] --> COMBOS[verbalise + dedupe combos]
        COMBOS -->|A: embed + load| SQLDB
        COMBOS -->|B: RowSourceWalker + row_mode build| RLAT
    end
```

One deliberate asymmetry: the **orchestrating UDF is shared**. It is Fabric-native
infrastructure (User Data Functions are a Fabric item), so Backend A remains "pure
Fabric" even though a UDF fronts it. The UDF exists because (a) the Rayfin app's
sanctioned data paths today are Execute-DAX-Queries and its own generated GraphQL —
direct SQL connectivity is "coming soon" — and (b) both backends need a server-side
place to embed queries and hold gating config. Each backend also remains independently
callable without the UDF (A: T-SQL proc; B: existing `slice_stream`).

---

## 6. Backend A — pure Fabric (SQL database in Fabric)

### 6.1 Capability basis (verified July 2026)

| Piece | Status | Notes |
|---|---|---|
| `VECTOR(n)` type (float32, ≤1998 dims) | **GA** in Fabric SQL DB | [vector-data-type](https://learn.microsoft.com/sql/t-sql/data-types/vector-data-type?view=fabric-sqldb) |
| `VECTOR_DISTANCE` ('cosine'\|'euclidean'\|'dot'), `VECTOR_NORMALIZE`, `VECTORPROPERTY` | **GA** | exact KNN via `ORDER BY`; fine to ~50k vectors per Microsoft guidance |
| `CREATE VECTOR INDEX` (DiskANN) + `VECTOR_SEARCH` | Preview | not needed at Contoso scale; **off by default** here (approximate ⇒ weaker determinism story, §13) |
| `CREATE EXTERNAL MODEL` + `AI_GENERATE_EMBEDDINGS` (+ `AI_GENERATE_CHUNKS`) | Preview (Nov 2025) | Azure OpenAI / OpenAI / Ollama endpoints; key or **managed identity** credential |
| `sp_invoke_external_rest_endpoint` | Available, on by default | 1–230s timeout, 100 MB payload; Fabric-specific destination allowlist undocumented |
| Billing | CU-based, auto-pause after 15 min idle | storage billed while paused; cold resume adds first-query latency |

### 6.2 Schema (`semslice`)

The vector store is a SQL database in Fabric — either a dedicated one or the Rayfin
app's own generated DB *plus* a manually-managed schema (the portal DB is
schema-read-only from the portal but code-defined schemas deploy via the app; keep it
**separate** to avoid coupling to Rayfin's codegen — decision: **dedicated DB**,
`contoso-semslice`).

```sql
CREATE SCHEMA semslice;

-- One pair of tables per encoded dim (product / customer / store).
CREATE TABLE semslice.combo_product (
    combo_id      int            NOT NULL PRIMARY KEY,   -- dense rank over combo_text
    combo_hash    binary(32)     NOT NULL UNIQUE,        -- sha256(combo_text)
    combo_text    nvarchar(1000) NOT NULL,               -- the receipt
    member_count  int            NOT NULL,
    vec           vector(1536)   NOT NULL                -- ada-002 / text-embedding-3-small
);

CREATE TABLE semslice.combo_product_member (
    combo_id    int NOT NULL REFERENCES semslice.combo_product (combo_id),
    product_key int NOT NULL,
    PRIMARY KEY (combo_id, product_key)
);

CREATE TABLE semslice.dim_config (
    dim           sysname       NOT NULL PRIMARY KEY,    -- 'product' | 'customer' | 'store'
    key_column    nvarchar(200) NOT NULL,                -- "'Product'[ProductKey]"
    floor_sim     float         NOT NULL,                -- gate: min best-similarity
    band          float         NOT NULL,                -- keep sim >= best - band
    abs_min       float         NOT NULL,
    max_combos    int           NOT NULL,
    max_keys      int           NOT NULL,
    template_ver  varchar(20)   NOT NULL,
    embed_model   varchar(100)  NOT NULL,                -- pins the deployment + version
    built_utc     datetime2     NOT NULL
);
```

Scale check: Contoso 100k has low-thousands of product combos, a few thousand customer
combos, dozens of store combos — exact KNN over ≤10k 1536-d vectors is
single-digit-milliseconds territory; no DiskANN, no index maintenance.

*Footnote on mirroring (correctly scoped):* tables containing `VECTOR` columns are
excluded from the SQL DB's automatic OneLake mirroring. **This design doesn't care** —
the `semslice` tables are operational-only and are queried directly over T-SQL; the
semantic model lives elsewhere (§3). It would only matter if someone chose to host the
star schema in this same DB *and* wanted Direct Lake over its mirror — in that layout,
keep vectors in separate tables (as this schema already does) and the dims mirror fine.

### 6.3 Embedding options (index-time and query-time must match)

| | Option A1 — BYO Azure OpenAI (recommended) | Option A2 — Fabric prebuilt AI only | Option A3 — rlat encoder vectors in SQL (bridge) |
|---|---|---|---|
| Model | `text-embedding-3-small` (1536) or ada-002, your AOAI resource | `text-embedding-ada-002` (1536) — the only embedding model Fabric hosts (preview) | `gte-modernbert-base` 768-d |
| Index-time | `AI_GENERATE_EMBEDDINGS(combo_text USE MODEL ContosoEmbedder)` in a set-based `UPDATE`, or notebook | notebook `ai.embed` / OpenAI SDK with Fabric identity, write vectors via JDBC/pyodbc | UDF `embed` batched from the build notebook |
| Query-time | same function inside the search proc — **one T-SQL round trip** | UDF embeds via prebuilt AOAI REST with Fabric token, passes `@qv` to proc (prebuilt-AI-from-UDF: **spike S3**) | proc calls UDF `embed` via `sp_invoke_external_rest_endpoint` (the existing `dbo.rlat_search` pattern) |
| Extra Azure resources | 1 (AOAI) | 0 | 0 |
| Notes | cleanest pure-SQL demo; managed-identity credential; preview surface | truest "nothing but Fabric" story; bills CU (3.36 CU-s/1k tokens) | not "without rlat" — it's the parity bridge; vector spaces identical to Backend B |

**Decision: A1 primary, A2 as the zero-extra-resource variant** (same schema, swap the
embedder). A3 stays documented because it already exists in spirit (`embed`'s docstring
carries the SQL envelope contract) and gives an apples-to-apples vector-space comparison
between backends when wanted.

External model + credential (A1):

```sql
CREATE DATABASE SCOPED CREDENTIAL [https://<aoai>.openai.azure.com]
WITH IDENTITY = 'Managed Identity',
     SECRET = '{"resourceid":"https://cognitiveservices.azure.com"}';

CREATE EXTERNAL MODEL ContosoEmbedder
WITH ( LOCATION = 'https://<aoai>.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2024-02-01',
       API_FORMAT = 'Azure OpenAI',
       MODEL_TYPE = EMBEDDINGS,
       MODEL = 'text-embedding-3-small',
       CREDENTIAL = [https://<aoai>.openai.azure.com] );
```

### 6.4 Search procedure

```sql
CREATE PROCEDURE semslice.search
    @query nvarchar(1000),
    @qv    vector(1536) = NULL   -- pre-embedded (A2/A3 paths); NULL → embed in-proc (A1)
AS
BEGIN
    SET NOCOUNT ON;
    IF @qv IS NULL
        SET @qv = AI_GENERATE_EMBEDDINGS(@query USE MODEL ContosoEmbedder);

    -- per dim: ranked combos (shown for product; repeated per dim or via dynamic SQL
    -- over dim_config)
    WITH ranked AS (
        SELECT TOP (50) c.combo_id, c.combo_text, c.member_count,
               1.0 - VECTOR_DISTANCE('cosine', @qv, c.vec) AS sim
        FROM semslice.combo_product AS c
        ORDER BY sim DESC, c.combo_id ASC          -- explicit deterministic tie-break
    )
    SELECT 'product' AS dim, r.combo_id, r.combo_text, r.sim, m.product_key
    FROM ranked AS r
    JOIN semslice.combo_product_member AS m ON m.combo_id = r.combo_id
    ORDER BY r.sim DESC, r.combo_id, m.product_key;
END
```

Gating (§4.3) is applied by the orchestrating UDF from `dim_config` (keeps thresholds
in one place for both backends). A pure-SQL variant (gate inside the proc) is trivial if
someone wants the no-UDF demo; both are deterministic.

Latency budget (warm): embed ~50–150 ms (AOAI) + 3 × exact KNN ≤ ~20 ms + expansion
join ≈ **sub-300 ms**. Cold (auto-paused DB): add resume time — for demos, keep the DB
warm with a scheduled trivial query, or accept and show the `cold` flag (both backends
have a cold story; surface it honestly).

---

## 7. Backend B — rlat

### 7.1 Build — one row-mode KM per dim

`fabric_build.ipynb` gains a **slicer build mode** (new work), or a sibling
`contoso_slicer_build.ipynb`:

```python
from resonance_lattice.build.walker import RowSourceWalker
from resonance_lattice.build.pipeline import build_rlat

for dim in ("product", "customer", "store"):
    combos = verbalise_and_dedupe(read_delta(f"contoso.dim_{dim}"))   # §4.1 templates
    walker = RowSourceWalker(
        rows=((f"{c.combo_id}", c.combo_text) for c in combos),
        source_name=f"contoso-{dim}",
    )
    build_rlat(walker, out=f"/lakehouse/default/Files/rlat/contoso-{dim}.rlat",
               row_mode=True)                      # bundled mode enforced by row_mode
    write_delta(f"semslice_{dim}_combo_member", combos.member_map)    # combo → keys
```

Decisions:
- **Per-dim `.rlat` files** (`contoso-product.rlat`, …) rather than one combined file
  with prefixed keys: independent refresh, per-dim gating maps 1:1 to per-KM calls, and
  discovery (`list_kms`) reads naturally. A combined file remains possible (keys like
  `product:4638`) but buys nothing here.
- **Keys are `combo_id`s, not member keys** — same dedupe rationale as §4.1. The
  combo→member expansion map lives as a small Delta table (or is folded into the UDF
  response by reading a parquet next to the KMs). This is a *change from v1*, where the
  key was the row's business key directly; `slice_stream` is unaffected (keys are opaque
  strings to it).
- Combo texts are tiny (≤1k chars), so bundled mode stays small; the receipts
  (`hits[].text`) come straight out of the archive — no join-back needed, unlike SQL.

### 7.2 Query — existing surface, new fan-out

`slice_stream(kmName, query, topK, snippetTopN)` already returns
`{keys, hits[{key, score, text}]}` OOM-safe over any corpus size. New: a thin
`slice_multi` UDF function that embeds once (one warm encoder), scans the three KMs'
bands sequentially (they share the process; per-KM mmap streaming already exists),
applies gating from a config file beside the KMs (`Files/rlat/contoso-slicer.json`,
mirroring `semslice.dim_config`), expands combo→members, and emits the §4.4 contract.

No AI service, no database: embeddings are computed by the in-process ONNX encoder,
vectors stream from the `.rlat` mmap. Warm query ≈ encode (~10–30 ms CPU) + 3 GEMVs
(ms) + expansion. Cold start pays the `.rlat` download + encoder load (seconds; the
`cold` flag reports it) — and **nothing runs or bills between queries**.

---

## 8. Shared orchestration — the `semantic_slice` UDF

One new function in `fabric/udf/function_app.py`:

```python
@udf.connection(alias="kmLakehouse", argName="lakehouse")
@udf.connection(alias="semsliceDb", argName="sqldb")     # Fabric SQL DB native connection
@udf.function()
def semantic_slice(lakehouse, sqldb, query: str, backend: str = "rlat") -> dict:
    """query + backend ('rlat'|'fabric-sql') → §4.4 response contract."""
```

- `backend="rlat"` → §7.2 path (encoder + KM streaming, gate, expand).
- `backend="fabric-sql"` → embed per §6.3 decision, `EXEC semslice.search @qv=…` over
  the native UDF↔SQL-DB connection, gate, emit.
- Both log one telemetry row per call (query, backend, per-dim best/gate, latencies) to
  the existing `udf_telemetry`-style Delta tables — this feeds threshold calibration
  and makes the side-by-side measurable, reusing `fabric_analytics.ipynb` patterns.
- Auth: the documented SPA flow (Entra app, delegated `UserDataFunction.Execute.All`,
  PKCE) — the Rayfin app's user token calls it. **CORS from the app origin = spike S2.**

Why the switch lives server-side: identical client code, honest latency comparison
(same hop count), and gating config never ships to the browser.

---

## 9. Semantic model & DAX injection

The app queries the Contoso model through the **Execute DAX Queries API** — the
sanctioned data path for Rayfin data apps (tenant setting on; Build+Read on the model).
Matched keys are injected as **literal set predicates**:

```dax
EVALUATE
CALCULATETABLE (
    SUMMARIZECOLUMNS (
        'Date'[Year Month],
        "Sales Amount", [Sales Amount],
        "Margin",       [Margin],
        "Orders",       [Order Count]
    ),
    'Product'[ProductKey]   IN {4638, 2113, 977},
    'Customer'[CustomerKey] IN {18023, 44107 /* … */}
    -- Store gate closed → no predicate; dim untouched
)
```

- **`IN` over `TREATAS` for literal lists** — same semantics, cheaper plan (SQLBI);
  TREATAS remains the tool if a filter table ever replaces literals.
- Every visual on the page re-issues its query with the same predicate block appended —
  the app template already centralises DAX generation per visual, so injection is one
  code path.
- Limits that matter: JSON API caps 100k rows / 15 MB / **120 queries per user per
  minute** — a page of ~6 visuals re-querying per utterance is well inside it. No
  documented cap on literal-list length; hundreds-to-low-thousands of keys is proven
  territory (v1 ran 200-key TREATAS lists). If a gate ever legitimately passes tens of
  thousands of keys, that's the §4.3 degrade-to-unfiltered case, not a bigger literal.
- **No write-back, no refresh, no reframing**: the filter exists only inside each query,
  so model storage mode (import vs Direct Lake) is orthogonal, and "filtered on the
  fly" is literal — next query, new world.

Fallback surface (kept in the design as Plan B if a Rayfin preview limitation bites):
a static SPA with **powerbi-client** embedding the standard Contoso report and applying
`IBasicFilter { operator: "In", values: keys }` via `report.updateFilters(...)` — same
backends, same contract, report-native visuals, filter pane shows the applied filters.

---

## 10. Front end — Fabric App (Rayfin, preview)

Scaffold: `npm create @microsoft/rayfin@latest -- "contoso-semantic-slicer"
--template dataapp --workspace <ws>` → TypeScript data model, generated SQL DB +
GraphQL, static SPA hosted at `https://<app>-app.rayfin.windows.net/`, Fabric SSO only,
deploy via `npx rayfin up`. Requirements: tenant setting **Fabric Apps (preview)** on,
F-capacity workspace, supported region.

Page layout (single page, the demo *is* the app):

1. **Ask bar** — text box + push-to-talk mic button (§11) + backend toggle
   (`rlat ⇄ fabric-sql`) + example-query chips.
2. **Applied-filters panel** — per dim: gate state (chip: *filtered / no match / too
   broad*), best score, member count, and the top receipts (combo text + score bars —
   the template's data grid with custom cell renderers is built for exactly this).
   This is the "what dim filters are applied" requirement, answered from the §4.4
   response, richer than a filter pane could be (scores + receipts).
3. **Visuals** — template-native components bound to injected DAX (§9): KPI cards
   (Sales, Margin, Orders — with unfiltered-baseline deltas, the v1 "£230 → £289"
   moment), a trend line, category/geography bars, and a matched-members grid.
4. **Latency strip** — embedMs / searchMs / DAX ms / cold flags per backend; the
   side-by-side receipt.

State model: `utterance → contract → predicates → parallel visual queries`; everything
re-renders from one state object, so voice and typed input are the same code path after
transcription.

Known preview risks (tracked in §16 spikes): apps connected to semantic models
currently **can't open outside the Fabric portal** (standalone window errors) — which
makes in-portal-iframe **microphone permission delegation the #1 spike**; direct
SQL-endpoint connectivity from apps is "coming soon" (hence the UDF hop); no secret
storage in the app (hence the token relay).

---

## 11. Voice input

Push-to-talk, transcribe-then-search (not streaming search-as-you-speak — one
utterance, one filter application, matching §4.2):

**Primary — Azure AI Speech, browser SDK (`microsoft-cognitiveservices-speech-sdk`):**

1. App (signed in via Fabric SSO) calls the **token relay** — a small UDF
   `speech_token()`: reads the Speech key from **Azure Key Vault via a UDF generic
   connection** (owner-identity authorised), POSTs to the regional
   `sts/v1.0/issueToken`, returns the **10-minute JWT** + region. No key ever reaches
   the browser; the app refreshes ~every 9 minutes. (Rayfin has no secret store yet —
   relay confirmed pattern. If UDF CORS blocks the browser (spike S2), the relay
   becomes a tiny Azure Function with managed identity.)
2. `SpeechConfig.fromAuthorizationToken(token, region)` +
   `AudioConfig.fromDefaultMicrophoneInput()` → **`recognizeOnceAsync()`** — purpose-built
   for command/query capture: streams mic audio over WebSocket, finalises at
   end-of-utterance/silence (≤30 s), interim hypotheses can render live in the ask bar.
3. Final transcript → the exact typed-query path.
4. Optional `AutoDetectSourceLanguageConfig` (≤4 locales) for multi-language demos.

Cost/limits: F0 = 5 free audio-hours/month (covers demos), S0 ≈ US$1/audio-hour.
Works in Edge/Chrome/Safari/Firefox (it's getUserMedia + WebSocket, not Web Speech
API).

**Demo-grade fallback — Web Speech API** (`webkitSpeechRecognition`): ~10 lines, zero
cost, zero auth — but Chrome-only in practice (Edge's implementation is unreliable,
Firefox disabled) and Chrome routes audio to Google's cloud — a data-governance
disclaimer, not a production option. Feature-detect; always keep the text box.

**Alternative (no streaming, simplest client)** — `MediaRecorder` (webm) → POST blob to
a relay (Azure Function, Entra-protected, managed identity) → Azure OpenAI
`gpt-4o-transcribe` / whisper `/audio/transcriptions` → text. Clean fit if an AOAI
resource already exists for Backend A1; no interim results.

**The iframe question (spike S1):** if the Fabric portal doesn't delegate
`allow="microphone"` to the app iframe, `getUserMedia` fails regardless of SDK. Escape
hatches, in order: (a) the standalone-window limitation is fixed and the app runs at
its own origin (expected eventually — it *is* a public URL today, just erroring on
model queries); (b) a companion capture page at the app origin opened in its own tab,
posting transcripts back via `BroadcastChannel`; (c) fall back to the powerbi-client
SPA (§9 Plan B) where the origin is ours. Voice inside a Power BI **custom visual** is
architecturally impossible (sandboxed iframe, no microphone privilege) and stays out.

---

## 12. Determinism & explainability

The claim "a natural language query **deterministically** filters the dims" decomposes
into three links, each with different strength per backend — stated honestly:

| Link | Backend A (SQL) | Backend B (rlat) |
|---|---|---|
| text → vector | AOAI embeddings: stable in practice, **no published bit-exactness guarantee**; pinned deployment; any model-version change ⇒ re-embed everything (`embed_model` column enforces the pairing) | **bit-exact on CPU** — tested guarantees D1–D3, revision-pinned encoder |
| vector → ranked combos | exact KNN (`ORDER BY VECTOR_DISTANCE … , combo_id`) — fully deterministic; DiskANN would break this (approximate), hence off | exact full-band GEMV, ranked `(cosine, key)` — fully deterministic |
| combos → filters | fixed thresholds from config, deterministic expansion join, `IN {…}` set semantics | same (shared gating config semantics) |

So: **given a fixed build and fixed config, the same query text always produces the same
filter set** — strictly true end-to-end on Backend B; true modulo the embedding
service's stability on Backend A. Nothing in the loop is an LLM making a judgement
call. That's the positioning against NL→DAX (§15): those tools can *answer questions*;
this filters **your** curated visuals, repeatably, with receipts.

And it stays a **suggester, not an oracle** (v1's honest boundary): the gate can be
wrong at the margins, thresholds are calibrated not divine, and structured constraints
belong to structured filters. The receipts panel exists so users can see — and
distrust — every match.

---

## 13. Backend comparison (what the demo should surface)

| | A — Fabric SQL vectors | B — rlat |
|---|---|---|
| Moving parts | SQL DB (+ AOAI unless A2), UDF, model, app | OneLake files, UDF, model, app |
| Embeddings | ada-002/3-small via service call, billed (CU or AOAI) | local ONNX in the UDF worker, free |
| Query-time deps | AI endpoint up + DB resumed | none beyond the UDF worker |
| Receipts | join back to `combo_text` (designed in — §6.2) | native (`hits[].text` from the archive) |
| Idle cost | DB storage while paused; resume latency | zero — files at rest |
| Determinism | exact KNN; embedding service "stable, unguaranteed" | bit-exact, tested |
| Refresh | re-run notebook → `UPDATE`/reload combos | re-run build → re-upload; UDF hot-swaps on mtime |
| Scale ceiling | GA engine; DiskANN preview available past ~50k vectors/dim | `slice_stream` proven at 94k rows; beyond that, band scan is linear |
| Fabric-purity | 100% Microsoft stack (A2 variant: zero non-Fabric resources) | one OSS library riding Fabric-native items |
| Preview surface | AI_GENERATE_EMBEDDINGS (preview); vectors GA | none (UDF + OneLake GA) |

The pitch writes itself both ways — A: "your platform already does this"; B: "one file,
no services, bit-exact." Showing both **is** the content.

---

## 14. Alternatives considered

- **Translytical task flows** (GA 03/2026) — button → UDF → write keys to a filter
  table → model picks it up. Rejected as primary: UDF return values cannot set report
  filters, so state flows through a table + refresh/reframing (~15 s mirroring + framing
  for Direct Lake) — versus zero-latency per-query injection. Retained as the pattern
  for *persisted/shared* filter states ("save this slice for the team").
- **powerbi-client embed SPA** — fully viable, GA, filter-pane visibility for free;
  demoted to Plan B only because the brief targets the Rayfin data app as the surface.
- **Copilot data questions / Fabric data agents / Fabric IQ** — answer questions by
  *generating* DAX/SQL; non-deterministic, don't drive existing curated visuals, no
  filter-state API. Complementary, not competing: nothing native does
  "NL/voice → deterministic filters on fixed visuals" as of July 2026.
- **Vectors in Eventhouse/KQL or a lakehouse + FAISS notebook** — both can nearest-
  neighbour, neither offers the transactional, T-SQL-proc-friendly, GA vector story SQL
  DB has; out of scope.

---

## 15. Delivery plan

**Phase 0 — spikes (order = risk):**
- **S1** Mic inside the Fabric App iframe (`getUserMedia` in-portal; test the
  standalone-window behaviour too). Fallbacks pre-designed (§11).
- **S2** Browser `fetch` → UDF public endpoint CORS with an Entra bearer from the app
  origin.
- **S3** (A2 only) Prebuilt-AOAI embedding call from a UDF with Fabric identity.
- **S4** `sp_invoke_external_rest_endpoint` → UDF endpoint reachability (A3 bridge
  only; allowlist undocumented).
- **S5** Verify Contoso V2 exact dim/column names against SQLBI docs; decide V2 vs
  ContosoRetailDW-shape for Promotion/Education richness.

**Phase 1 — data + builds:** lakehouse load (SQLBI Delta); verbalisation module
(templates §4.1, versioned, shared by both backends — one Python module, unit-tested
dedupe/ordering); `semslice` schema + embed + load (A); per-dim row-mode `.rlat` builds
(B); combo→member maps.

**Phase 2 — query services:** `semslice.search` proc; `slice_multi`/gating; the shared
`semantic_slice` UDF + telemetry; gating-config files; parity harness (same 50 queries
→ both backends → report per-dim gate agreement and key-set Jaccard — not expected to
match across different embedding spaces, but the *gate decisions* should mostly agree;
disagreements are calibration input).

**Phase 3 — model + app:** publish Contoso model (import first, Direct Lake variant
second); scaffold Rayfin data app; DAX injection layer; filter panel + visuals +
latency strip; backend toggle.

**Phase 4 — voice:** Key Vault + `speech_token` relay UDF; Speech SDK push-to-talk;
Web Speech fallback behind feature detection; iframe escape hatch per S1 outcome.

**Phase 5 — calibrate + tell the story:** labelled query set (~50 utterances across
mono-dim / multi-dim / no-dim cases); tune `floor/band` per dim per backend from
telemetry; demo script (the £230→£289 moment, Contoso edition: *"show me premium silver
audio products bought by young professionals in Sydney"* → three gates open, margin
jumps, receipts on screen); write-up (sequel to the v1 post).

---

## 16. Risks & open questions

| # | Risk | Exposure | Mitigation |
|---|---|---|---|
| R1 | Fabric Apps preview churn (can't-open-standalone; SQL connectivity "coming soon"; GA date unstated) | front end | Plan B embed SPA shares every layer below the UI |
| R2 | Mic permission in portal iframe (UNCONFIRMED) | voice | S1 first; escape hatches §11 |
| R3 | UDF CORS from app origin (UNCONFIRMED) | both backends' client path | S2; relay/proxy via Azure Function if needed |
| R4 | `AI_GENERATE_EMBEDDINGS` preview behaviour/regions | Backend A1 | A2 notebook-embeds at build + UDF at query; or plain REST from UDF |
| R5 | Embedding-model version drift silently shifts A's vector space | Backend A | `embed_model` pinned per table; rebuild-on-change rule; parity harness catches gross shifts |
| R6 | Whole-utterance matching cross-contaminates dims (e.g. city names in product text) | quality | gating floors + per-dim templates with role labels; calibration set; degrade-to-unfiltered |
| R7 | 120 executeQueries/min/user | busy pages | batch visuals per utterance (one EVALUATE per visual, ≤10/utterance); debounce |
| R8 | SQL DB auto-pause cold hits mid-demo | Backend A | keep-warm ping or narrate the `cold` flag honestly (B has the same story at file level) |
| Q1 | Are `slice_multi` + expansion better in-UDF or is per-KM `slice_stream` × 3 from the app acceptable? | latency vs simplicity | measure in Phase 2; contract §4.4 is unchanged either way |
| Q2 | Date/Promotion dims: add ContosoRetailDW-shape data for a richer story? | scope | decide after S5 |

---

## 17. Sources

Fabric SQL DB vectors: [vector type](https://learn.microsoft.com/sql/t-sql/data-types/vector-data-type?view=fabric-sqldb) ·
[VECTOR_DISTANCE](https://learn.microsoft.com/sql/t-sql/functions/vector-distance-transact-sql?view=sql-server-ver17) ·
[vector indexes / VECTOR_SEARCH (preview)](https://learn.microsoft.com/sql/t-sql/functions/vector-search-transact-sql?view=fabric-sqldb) ·
[CREATE EXTERNAL MODEL](https://learn.microsoft.com/sql/t-sql/statements/create-external-model-transact-sql?view=sql-server-ver17) ·
[AI_GENERATE_EMBEDDINGS](https://learn.microsoft.com/sql/t-sql/functions/ai-generate-embeddings-transact-sql?view=sql-server-ver17) ·
[sp_invoke_external_rest_endpoint](https://learn.microsoft.com/sql/relational-databases/system-stored-procedures/sp-invoke-external-rest-endpoint-transact-sql?view=fabric-sqldb) ·
[AI use case](https://learn.microsoft.com/fabric/database/sql/use-case-ai-application) ·
[mirroring limitations](https://learn.microsoft.com/fabric/database/sql/mirroring-limitations) ·
[billing](https://learn.microsoft.com/fabric/database/sql/usage-reporting) ·
[prebuilt AI services](https://learn.microsoft.com/fabric/data-science/ai-services/ai-services-overview)

Fabric Apps / Rayfin: [overview](https://learn.microsoft.com/fabric/apps/overview) ·
[data app template](https://learn.microsoft.com/fabric/apps/data-apps-template) ·
[analytics with Fabric Apps](https://learn.microsoft.com/power-bi/create-reports/fabric-apps-analytics) ·
[Rayfin SDK](https://learn.microsoft.com/javascript/api/fabric-apps-sdk-javascript/rayfin-overview) ·
[pricing](https://learn.microsoft.com/fabric/apps/pricing) ·
[announcement](https://community.fabric.microsoft.com/t5/Fabric-Updates-Blog/Introducing-Rayfin-A-new-AI-first-way-to-build-deploy-and-govern/ba-p/5191676)

DAX / model: [Execute DAX Queries (JSON)](https://learn.microsoft.com/rest/api/power-bi/datasets/execute-queries) ·
[Arrow variant + limits](https://learn.microsoft.com/power-bi/developer/execute-dax-queries-arrow/overview) ·
[SQLBI on IN vs TREATAS](https://www.sqlbi.com/blog/marco/2017/11/25/using-treatas-in-place-of-in-in-dax/) ·
[Direct Lake overview](https://learn.microsoft.com/fabric/fundamentals/direct-lake-overview) ·
[translytical task flows](https://learn.microsoft.com/power-bi/create-reports/translytical-task-flow-overview) ·
[control report filters (embed fallback)](https://learn.microsoft.com/javascript/api/overview/powerbi/control-report-filters)

Voice: [Speech SDK STT quickstart](https://learn.microsoft.com/azure/ai-services/speech-service/get-started-speech-to-text) ·
[token auth](https://learn.microsoft.com/azure/ai-services/authentication#authenticate-with-an-access-token) ·
[Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API) ·
[AOAI transcriptions](https://learn.microsoft.com/azure/foundry/openai/whisper-quickstart) ·
[UDF service limits](https://learn.microsoft.com/fabric/data-engineering/user-data-functions/user-data-functions-service-limits) ·
[UDF from SPA tutorial](https://learn.microsoft.com/fabric/data-engineering/user-data-functions/tutorial-invoke-from-python-app)

Data: [SQLBI Contoso Generator V2](https://www.sqlbi.com/tools/contoso-data-generator/) ·
[ready-to-use datasets](https://github.com/sql-bi/Contoso-Data-Generator-V2-Data/releases)

Repo: `fabric/udf/function_app.py` · `src/resonance_lattice/build/{walker,pipeline}.py` ·
`docs/site/semantic-slicer.html` (v1 write-up) · `docs/site/fabric.html` ·
`tests/harness/encoder_determinism.py`
