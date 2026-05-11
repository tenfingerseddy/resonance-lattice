# Resonance Lattice on Microsoft Fabric

A team-shared knowledge model. One `.rlat` file in OneLake, exposed as a Fabric User Data Function. Anyone on the workspace queries it from the CLI, Claude Code, or a Fabric notebook — no per-user encoder install, no `.rlat` copies on each laptop.

The demo notebook (`fabric_demo_udf.ipynb`) queries the UDF and writes telemetry to two Delta tables. A second notebook (`fabric_analytics.ipynb`) reads those tables and deploys a Direct Lake semantic model you can point Power BI at.

## What you'll need

- A Fabric workspace, with a Lakehouse you can write `Files/` to
- ~500 MB of OneLake space (your `.rlat` plus telemetry)
- An Entra account on the workspace (signs you in once via device code)
- Power BI Desktop (optional — only if you want to build visuals on the semantic model)

The whole thing takes about 15 minutes the first time.

## The path

```
Files/<SRC>/  ──fabric_build.ipynb──►  Files/rlat/<KM>.rlat ──UDF.search──►  results
                                                                │
                                                                ▼
                                       fabric_demo_udf.ipynb ──► udf_telemetry + udf_hits Delta
                                                                       │
                                                                       ▼
                                                      fabric_analytics.ipynb
                                                                       │
                                                                       ▼
                                                  rlat-analytics semantic model
```

## 1. Build a `.rlat`

You have three options.

**A. Build inside Fabric (recommended for ongoing rebuilds).** Drop source files into `Files/<SOURCE_DIR>/` in your OneLake, then run [`notebooks/examples/fabric_build.ipynb`](../../notebooks/examples/fabric_build.ipynb). First run builds, every run after refreshes incrementally — only changed files are re-encoded. Schedule from a Fabric Pipeline if you want it on a cron.

**B. Build locally and upload.** If you'd rather build from your machine:

```bash
rlat build ./your-docs -o team-docs.rlat
rlat fabric publish-rlat ./team-docs.rlat \
    --workspace <workspace-id> \
    --lakehouse <lakehouse-id>
```

**C. Build on a GPU for big seed corpora.** For first-time builds over ~10k passages, Fabric CPU is slow. Build on Kaggle, Colab, or a local GPU machine, upload the seed `.rlat` to `Files/rlat/`, and let `fabric_build.ipynb` handle subsequent refreshes inside Fabric.

Either way, the `.rlat` ends up at `Files/rlat/<KM_NAME>.rlat`. Discovery is dynamic — the UDF lists `Files/rlat/` live, so any upload (notebook, helper, drag-drop) shows up on the next `list_kms` call.

If your workspace uses a different folder convention (`Files/knowledge/`, `Files/data/`), set `RLAT_FABRIC_KM_DIR=<dirname>` on the UDF item later. Default is `rlat`.

## 2. Create a User Data Function item

In the Fabric portal: **New → User Data Function**. Pick a name (e.g. `rlat-search`).

Open the item, then **Manage Connections → add Lakehouse**. Pick the lakehouse from step 1 and set its alias to **`kmLakehouse`** — that's the alias the UDF code references.

## 3. Add the `rlat` library and publish the code

Still on the UDF item: **Library Management → Public libraries → Add from PyPI**, search for `rlat`, install at workspace scope. Pin a version (`==2.1.0a13` or whatever's current).

Then publish the code from this repo's [`fabric/udf/`](../../fabric/udf/) directory. The simplest path: open `fabric/udf/` in VS Code with the Microsoft Fabric extension, sign in, map to your UDF item, and Publish. The directory holds:

```
fabric/udf/
  function_app.py    # search + list_kms entry points
  requirements.txt   # local-emulator fallback
```

## 4. Test the UDF

In the UDF item, select `search` → set `kmName=<your km>`, fill in `query`, hit **Run**. You should get back JSON like `{band, cold, hits[]}`. First call may take a few seconds (cold load); subsequent calls are sub-second.

If you see `could not stat Files/rlat/<km>.rlat`, the upload path or the `kmLakehouse` alias isn't matching. Check the upload location and Manage Connections.

## 5. Query from your machine

```bash
pip install 'rlat[fabric]'
rlat fabric add team=https://<your-udf-endpoint-url>
```

The endpoint URL is on the UDF item under **Properties → Endpoint URL**. The `add` command writes the alias to `~/.config/rlat/fabric.toml`. (If you're using Claude Code, it also scaffolds a `.claude/skills/rlat-fabric-search/` entry so the agent sees it as a tool.)

```bash
rlat search fabric://team                          # list available KMs
rlat search fabric://team/team-docs "how do I deploy?"
rlat search fabric://team/team-docs "..." --format=context
```

The first call prompts an Entra device-code sign-in (URL + code printed to stderr). Sign in once; the token caches to your OS keyring and subsequent calls are silent.

## 6. Run the demo notebook (writes telemetry)

Upload [`notebooks/examples/fabric_demo_udf.ipynb`](../../notebooks/examples/fabric_demo_udf.ipynb) into your Fabric workspace and open it. Set:

- `UDF_BASE_URL` to your UDF's endpoint URL (the part before `/functions/<name>/invoke`)
- `KM_NAME` to your KM (e.g. `team-docs`)
- `DEMO_QUERIES` to a list of queries you want logged

Run it. You'll see retrieval results inline, plus two Delta tables appear in the bound lakehouse:

- `udf_telemetry` — one row per query (latency, top-1 score, cold/warm flag, refusal flag)
- `udf_hits` — one row per retrieved passage (rank, source file, drift status)

Re-run any time you want a new batch of telemetry. Both tables are append-only.

## 7. Run the analytics notebook (deploys the semantic model)

Upload [`notebooks/examples/fabric_analytics.ipynb`](../../notebooks/examples/fabric_analytics.ipynb). Set `UDF_BASE_URL` to the same endpoint, leave `SM_NAME` as `rlat-analytics`, run.

The notebook reads `udf_telemetry`, `udf_hits`, and the per-build JSON files under `Files/.rlat-builds/`. It materialises `udf_builds`, rebuilds four conformed dimension tables (`dim_query`, `dim_source_file`, `dim_km`, `dim_date`), and deploys a Direct Lake semantic model `rlat-analytics` to your workspace.

Idempotent — re-run after every batch of demo runs or build invocations, or schedule it via a Fabric pipeline. Dims rebuild from facts each time.

## 8. (Optional) Connect Power BI

Open Power BI Desktop → **Get Data → Power BI semantic models → `rlat-analytics`**. The semantic model ships with measures grouped under `01 Activity`, `02 Quality`, `03 Drift`, `04 Latency`, `05 Coverage`, and `06 Maintenance`, joined to `dim_query`, `dim_source_file`, `dim_km`, and `dim_date`. Drop the measures and dims on the canvas to build whatever views your team cares about — top queries, refusal rate, latency percentiles, build cadence.

That's the whole loop.

## How it stays fresh

The UDF caches `(.rlat contents, store, encoder)` per `(kmName, OneLake last_modified)`. Every search call stats the OneLake `.rlat` (cheap); when its `last_modified` changes, the cache evicts and rebuilds from the new bytes on the next call. Re-upload the `.rlat` after a `rlat build` / `rlat refresh` (or after the build notebook runs) and the next query sees fresh results — no UDF redeploy.

Up to 8 distinct `(kmName, mtime)` tuples are cached at once; older entries fall out (LRU).

## Cold-start latency

First call after a UDF container recycle pays:

- `.rlat` download from OneLake (~100–300 MB)
- Encoder ONNX cache from HuggingFace (~250 MB, revision-pinned)
- ONNX runtime warm + first encode

Roughly 5–15s cold, sub-100ms warm. The UDF returns a `cold` flag so callers can tell which they paid for.

## Auth

Two paths, in priority:

1. **Service-principal env vars** — if `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID` are all set, a `ClientSecretCredential` is used silently. Right for CI or shared machines.
2. **Device-code flow** (default) — `rlat search` prints a URL and code to stderr. Open the URL, paste the code. Token caches to your OS keyring (libsecret / Keychain / Credential Manager). Works inside Claude Code: the prompt comes back through the Bash tool's stdout, you complete the browser flow, and the agent's tool call resolves with the search results.

Token cache name: `rlat-fabric`. Scope: `https://analysis.windows.net/powerbi/api/.default` (the Power BI scope, used to invoke UDFs — not the Fabric API scope, which is for item CRUD).

## Encoder distribution

Each `.rlat` records the encoder revision (HuggingFace commit SHA) it was built with. At UDF cold-start, the runtime reads that revision from the `.rlat` metadata and downloads the matching ONNX cache from `tenfingers/rlat-gte-modernbert-base-onnx`, revision-pinned. No per-workspace encoder upload.

If your tenant blocks egress to huggingface.co, set `RLAT_FABRIC_ENCODER_SOURCE=onelake` on the UDF environment and stage the encoder cache to `Files/rlat-encoders/<revision>.tar.zst` instead.

## What's server-side vs client-side

| Task | Server (UDF) | Client (notebook / CLI) |
|---|---|---|
| Single-shot retrieval | yes | calls into UDF |
| Discovery (list KMs) | yes | calls into UDF |
| Build / incremental refresh | — | yes (`fabric_build.ipynb` in Fabric, or `rlat build` locally + `rlat fabric publish-rlat`) |
| Multi-hop deep-search | — | yes (drives the loop, calls UDF for each hop) |
| `rlat optimise` (MRL training) | — | yes (build + train locally, re-upload result) |
| Compare between hosted KMs | — | not in v1 — both KMs need to be local |

## Troubleshooting

- `fabric alias 'team' not registered` → run `rlat fabric add team=<url>`. `rlat fabric list` shows what's registered.
- Device-code prompt never resolves → token cache is stuck. Delete the `rlat-fabric` entry in your OS keyring and retry. Or switch to SP env vars for one-shot use.
- `could not stat Files/rlat/<km>.rlat` → either the `.rlat` is missing from OneLake at the expected path, or the Lakehouse alias `kmLakehouse` isn't bound on the UDF item.
- `km has empty backbone.revision in metadata` → the `.rlat` was built with an older rlat that didn't pin the encoder revision. Rebuild with rlat ≥ 2.0.
- HuggingFace 404 / unfamiliar encoder errors → the revision pinned in the `.rlat` doesn't have a matching commit in `tenfingers/rlat-gte-modernbert-base-onnx`. Either rebuild the `.rlat` against a published revision, or push the encoder cache to that repo at the right revision.
- The new alpha won't show up in Library Management → Fabric's PyPI metadata cache lags. Either wait, or upload the wheel directly via **Library Management → Private libraries**.

## See also

- [`STORAGE_MODES.md`](STORAGE_MODES.md#hosted-fabric-udf) — where Fabric hosting fits among bundled / local / remote.
- [`CLI.md`](CLI.md) — `fabric://` URL form and the `rlat fabric add/list/remove/publish-rlat` subcommands.
- [`notebooks/examples/fabric_build.ipynb`](../../notebooks/examples/fabric_build.ipynb) — runs the build/refresh against the OneLake filesystem mount. Schedulable from a Fabric Pipeline.
- [`notebooks/examples/fabric_demo_udf.ipynb`](../../notebooks/examples/fabric_demo_udf.ipynb) — Fabric notebook querying the UDF and writing the two Delta tables.
- [`notebooks/examples/fabric_analytics.ipynb`](../../notebooks/examples/fabric_analytics.ipynb) — deploys the Direct Lake semantic model.
