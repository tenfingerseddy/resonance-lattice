---
name: rlat-build-on-kaggle
description: >-
  Build large rlat knowledge models (.rlat files) on Kaggle's free T4 GPU when
  the local machine has no GPU or encoding would take too long on CPU. Walks
  the user through Kaggle account + CLI setup, builds a kernel script that
  installs rlat[build,ann], sparse-clones (or uploads) source, encodes the
  corpus on T4, writes a remote-mode .rlat back, and pulls the artefact home.
  Trigger when the user mentions Kaggle, "free GPU", "build remotely", "no
  GPU here", a corpus too large to encode locally (~10K+ passages on CPU),
  or wants to rebuild several .rlat files in one batch. Not for: small
  corpora that finish in minutes on CPU; users who already have a CUDA GPU
  (just `pip install rlat[build]` and run `rlat build` locally).
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# rlat-build-on-kaggle — encode large corpora on Kaggle's free T4

You are helping the user build a `.rlat` knowledge model on Kaggle's free T4
GPU. The flow is: **set up the kaggle CLI → write a kernel script that
installs rlat + builds the corpus on /kaggle/working → push → poll → pull
the .rlat back**. The fast path takes ~20-40 minutes wall time for a corpus
of ~50K passages.

## When to use this skill (and when not to)

| Local situation | Use this skill? |
|---|---|
| No GPU; corpus has >10K passages | **Yes** — encoding 10K passages on CPU runs ~30-60 min; T4 finishes in 2-5 min |
| Several `.rlat` to rebuild as a batch | **Yes** — one Kaggle session can build all of them |
| You want a remote-mode `.rlat` (consumers fetch source from a URL) and the source already lives on GitHub | **Yes** — Kaggle has fast access to GitHub + the GPU |
| You have a CUDA GPU on your machine | **No** — `pip install rlat[build]` then `rlat build … --runtime torch` is faster than the round-trip to Kaggle |
| Corpus has <2K passages | **No** — finishes in <2 min on CPU; the Kaggle round-trip is more overhead than the encode |
| You need to keep the corpus private and Kaggle internet egress is unacceptable | **No** — Kaggle kernels need internet on for `pip install`; sensitive corpora should be encoded locally or on a private GPU |

A T4 encodes `gte-modernbert-base` at roughly **1500-2500 passages/sec** at
batch size 64. Locally with no GPU and ONNX-CPU runtime, the same encoder
runs at roughly **80-150 passages/sec** depending on the CPU. The crossover
where Kaggle wins is usually around 5-10K passages once you account for the
~5-min push + queue + setup overhead.

## Step 1 — One-time setup (≈10 minutes)

The user needs three things before the first push: a Kaggle account, the
`kaggle` Python CLI, and an API token. Walk them through whichever pieces
they don't already have.

### Kaggle account
Sign up at [kaggle.com](https://www.kaggle.com/account/login) — free. Verify
the phone number under *Settings → Phone verification* — **GPU minutes are
locked behind phone verification**.

### CLI install
```bash
pip install kaggle
kaggle --version  # confirm install
```

### API token
1. Visit [kaggle.com/settings/account](https://www.kaggle.com/settings/account).
2. Click **Create New Token** under *API*. Browser downloads `kaggle.json`.
3. Move it to the location the CLI expects:
   - **Linux/macOS**: `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
   - **Windows**: `move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\kaggle.json`
4. Test:
   ```bash
   kaggle kernels list -m  # lists your own kernels (should not error)
   ```

### Windows: set `PYTHONUTF8=1` (one-time, very important)

The Kaggle Windows CLI decodes server responses with CP1252. Any non-ASCII
character in the response (em-dashes in your kernel title, Unicode in
server messages) crashes the response parser, and the crash *silently eats
a successful push* — the kernel exists server-side but the slug is never
recorded client-side. Symptoms: `'charmap' codec can't decode byte ...`
traceback, then `403 Permission denied` on every subsequent `status` /
`output` call.

Fix it durably:

```cmd
setx PYTHONUTF8 1   # cmd.exe
```

Then **open a new terminal**. From this point on, every Python process on
Windows runs in UTF-8 mode and the kaggle CLI works.

If you can't `setx` (locked-down corp machine), prefix every kaggle CLI
call with `PYTHONUTF8=1`:

```bash
PYTHONUTF8=1 kaggle kernels push -p kaggle/job --accelerator NvidiaTeslaT4
PYTHONUTF8=1 kaggle kernels status <username>/<slug>
PYTHONUTF8=1 kaggle kernels output <username>/<slug> -p ./outputs/
```

## Step 2 — Pick the source-storage pattern

Two patterns. Pick whichever matches where the source corpus lives.

### Pattern A — Source on GitHub (most common for public docs)

The kernel sparse-clones the relevant subdir at a pinned commit, builds a
**remote-mode** `.rlat` whose `manifest.json` records the URL pattern
`https://raw.githubusercontent.com/<repo>/<sha>/<scope>/...`, and consumers
fetch source on demand with SHA verification. The `.rlat` itself stays
small (just bands + ANN index + manifest).

This is the right pattern when:
- The corpus already lives in a public GitHub repo
- You want the `.rlat` to stay small for shipping (HF Hub, package, etc.)
- You're OK with consumers fetching source over the network at query time

### Pattern B — Local source uploaded as a Kaggle dataset

If the source isn't on GitHub (private codebase, scraped corpus,
internal docs), package it as a tarball, push to Kaggle as a private
dataset, then mount it in the kernel and build a **bundled-mode** `.rlat`.
The bundled mode embeds source bytes inside the .rlat; the consumer needs
nothing else.

```bash
# Local: package + push as dataset
tar czf my_corpus.tar.gz -C /path/to/corpus .
mkdir kaggle/data
mv my_corpus.tar.gz kaggle/data/
cat > kaggle/data/dataset-metadata.json <<EOF
{
  "title": "My corpus for rlat build",
  "id": "<your-username>/my-corpus-source",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
kaggle datasets create -p kaggle/data    # first time
# subsequent updates: kaggle datasets version -p kaggle/data -m "update"
```

Then in the kernel script (Step 3), mount the dataset by adding it to
`dataset_sources` in the kernel metadata, glob-discover the unpacked
files at `/kaggle/input/` (Kaggle auto-extracts `.tar.gz`), and run
`rlat build` against the discovered path with `--store-mode bundled`.

The rest of this skill walks Pattern A. Pattern B differs only in the
source-prep block; everything from "push the kernel" onwards is identical.

## Step 3 — Write the kernel script

Create a project-relative working dir (not `mktemp` — the Kaggle CLI has
path issues with temp dirs on Windows), then drop two files:
`build_corpus.py` (the work) and `kernel-metadata.json` (Kaggle config).

```bash
mkdir -p kaggle/rlat-build
```

**Pre-built templates ship with this skill** — copy them straight from
[`scripts/`](scripts/) and edit the `CONFIG` block:

```bash
# single-corpus build
cp .claude/skills/rlat-build-on-kaggle/scripts/build_corpus.py        kaggle/rlat-build/
cp .claude/skills/rlat-build-on-kaggle/scripts/kernel-metadata.json   kaggle/rlat-build/

# multi-corpus batch (resilient checkpointing — see "Multi-corpus batches" below)
cp .claude/skills/rlat-build-on-kaggle/scripts/build_corpora_batch.py kaggle/rlat-build/
```

Then edit the `CONFIG` block at the top of the script and the `id` /
`title` fields in the metadata JSON. Both files are self-contained and
have inline comments explaining each knob.

The full single-corpus template is reproduced below so you can read it
without leaving this skill — but if you're going to actually push, copy
the file from `scripts/` rather than re-typing.

### `kaggle/rlat-build/build_corpus.py`

Adapt the variables in the `CONFIG` block to your corpus.

```python
"""Build an rlat knowledge model on a Kaggle T4 in remote-mode.

Writes /kaggle/working/<name>.rlat as soon as the build finishes; the
file ships out at COMPLETE / ERROR.
"""
from __future__ import annotations
import json, shutil, subprocess, sys, time, traceback
from pathlib import Path

# ── Config: adapt to your corpus ────────────────────────────────────
CONFIG = {
    "name": "my-docs",                          # output: <name>.rlat
    "github": "owner/repo",                     # public GitHub repo
    "branch": "main",                           # branch to clone
    "scope": "docs",                            # subdir inside the repo to index
}
# ────────────────────────────────────────────────────────────────────

WORK = Path("/kaggle/working")
WORK.mkdir(exist_ok=True, parents=True)
SRC = WORK / "_src"
SRC.mkdir(exist_ok=True, parents=True)


def shell(cmd, log=None):
    print(f"$ {' '.join(cmd)}", flush=True)
    if log is None:
        return subprocess.run(cmd, check=True)
    with open(log, "ab") as f:
        f.write(f"\n$ {' '.join(cmd)}\n".encode())
        f.flush()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=False)
        f.write(proc.stdout)
    sys.stdout.write(proc.stdout.decode("utf-8", errors="replace"))
    sys.stdout.flush()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc


def main() -> int:
    # 1. Install rlat with the GPU build extras AND the ANN extras.
    #    `[ann]` pulls faiss-cpu, which rlat needs for HNSW above ~5000
    #    passages. Forgetting it lets the encode finish then crashes with
    #    "RuntimeError: faiss is not installed".
    shell([sys.executable, "-m", "pip", "install", "--quiet",
           "rlat[build,ann]"])
    shell(["rlat", "install-encoder"])

    # 2. Sanity-print: rlat version, encoder + pinned revision, CUDA presence.
    #    Failing these early saves a 30-minute encode against a broken env.
    shell([sys.executable, "-c",
           "import resonance_lattice as rl; print('rlat', rl.__version__);"
           "from resonance_lattice.install.encoder import MODEL_ID, PINNED_REVISION;"
           "print('encoder', MODEL_ID, '@', PINNED_REVISION);"
           "import torch; print('cuda:', torch.cuda.is_available());"
           "print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"])

    # 3. Sparse-clone the scope subdir at branch HEAD. Records the SHA so
    #    the remote-mode .rlat manifest can pin to it.
    target = SRC / CONFIG["github"].replace("/", "_")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    shell(["git", "init", "-q", "-b", CONFIG["branch"], str(target)])
    shell(["git", "-C", str(target), "remote", "add", "origin",
           f"https://github.com/{CONFIG['github']}.git"])
    shell(["git", "-C", str(target), "config", "core.sparseCheckout", "true"])
    (target / ".git/info/sparse-checkout").write_text(f"{CONFIG['scope']}/*\n")
    shell(["git", "-C", str(target), "fetch", "--depth", "1", "origin", CONFIG["branch"]])
    shell(["git", "-C", str(target), "checkout", "FETCH_HEAD"])
    sha = subprocess.check_output(
        ["git", "-C", str(target), "rev-parse", "FETCH_HEAD"], text=True).strip()
    print(f"pinned to {CONFIG['github']}@{sha}", flush=True)

    scope_dir = target / CONFIG["scope"]
    if not scope_dir.exists():
        raise RuntimeError(f"scope path {CONFIG['scope']!r} not found in clone")

    # 4. Build the .rlat in remote mode. The manifest records this URL
    #    pattern; consumers fetch source from raw.githubusercontent.com
    #    and SHA-verify it. --runtime torch picks CUDA on T4 (~10-30x
    #    faster than ONNX-CPU). --batch-size 64 doubles throughput at
    #    768d vs the default 32 without OOM on T4 (~13 GB RAM).
    out = WORK / f"{CONFIG['name']}.rlat"
    url_base = (f"https://raw.githubusercontent.com/{CONFIG['github']}"
                f"/{sha}/{CONFIG['scope']}/")
    log = WORK / f"{CONFIG['name']}.build.log"
    t0 = time.time()
    shell(["rlat", "build", str(scope_dir),
           "-o", str(out),
           "--store-mode", "remote",
           "--remote-url-base", url_base,
           "--runtime", "torch",
           "--batch-size", "64"], log=log)

    # 5. Write a result summary so the slug status panel + JSON record
    #    both report success cleanly.
    result = {
        "status": "ok",
        "rlat_path": str(out),
        "size_bytes": out.stat().st_size,
        "github": CONFIG["github"],
        "scope": CONFIG["scope"],
        "commit_sha": sha,
        "remote_url_base": url_base,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (WORK / "build_results.json").write_text(json.dumps(result, indent=2))
    print(f"DONE  size={result['size_bytes']:,} bytes  wall={result['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # Persist failure record so build_results.json is downloadable
        (WORK / "build_results.json").write_text(json.dumps(
            {"status": "failed", "error": f"{type(e).__name__}: {e}"}, indent=2))
        traceback.print_exc()
        sys.exit(1)
```

### `kaggle/rlat-build/kernel-metadata.json`

```json
{
  "id": "<your-username>/rlat-build-<corpus-name>",
  "title": "rlat build <corpus-name>",
  "code_file": "build_corpus.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_tpu": "false",
  "enable_internet": "true",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

**The slug must equal `slugify(title)`.** Kaggle's Save endpoint enforces
`id == slugify(title)` for *existing* kernels. If they don't match,
re-pushes return `409 Conflict`. Easiest fix: make both sides simple,
identical, all-lowercase-and-hyphens (e.g. `rlat-build-my-docs`).

Boolean fields are **strings** (`"true"` / `"false"`), not native JSON
booleans. The CLI rejects native booleans silently.

## Step 4 — Push the kernel

```bash
# Linux/macOS
kaggle kernels push -p kaggle/rlat-build --accelerator NvidiaTeslaT4

# Windows (no setx PYTHONUTF8 done yet)
PYTHONUTF8=1 kaggle kernels push -p kaggle/rlat-build --accelerator NvidiaTeslaT4
```

**Always pass `--accelerator NvidiaTeslaT4`.** The `enable_gpu` metadata
flag alone may assign a P100, which is sm_60 — incompatible with
contemporary PyTorch (sm_70+ minimum). CUDA ops fail silently on P100,
the encode falls back to CPU at roughly 1/30th speed, and the kernel
times out before finishing.

The push prints a kernel URL like `https://www.kaggle.com/code/<username>/<slug>`.
Save it — that's where you'll watch progress.

The "Your kernel title does not resolve to the specified id" line is a
**warning, not an error.** It fires when `id` and `slugify(title)`
disagree. On first push, Kaggle creates the kernel under the title slug,
so subsequent `status` calls against your `id` field return 403. **Make
`id` and `slugify(title)` match from the start** to avoid this.

## Step 5 — Poll for completion

```bash
kaggle kernels status <username>/<slug>
```

Status flow: `QUEUED` → `RUNNING` → `COMPLETE` (or `ERROR`).

The CLI doesn't stream progress. The web UI **does** — open
`https://www.kaggle.com/code/<username>/<slug>` and watch the "Log
Message" panel. While `RUNNING`:

- The web UI shows "Output 0 B" until the run terminates. **This is not
  a failure indicator** — Kaggle only ships `/kaggle/working/` to the
  output bucket at COMPLETE/ERROR. Files are being written; you just
  can't see them until the run ends.
- `kaggle kernels output …` mid-run returns the (often empty)
  auto-generated `.log` and **no** files from `/kaggle/working/`.

If `RUNNING` looks stuck for >10 min with no progress in the web UI's
log panel, the encode is genuinely hung. Cancel by re-pushing the same
slug — there's no `kaggle kernels stop` command, but a re-push
supersedes the running version.

If `QUEUED` for >5 min, GPU resources are contended; either wait
(usually clears in 10-30 min) or push at off-peak hours.

## Step 6 — Pull the .rlat back

```bash
kaggle kernels output <username>/<slug> -p ./outputs/ -o
```

This pulls everything in `/kaggle/working/` — the `.rlat`, the
`.build.log`, and `build_results.json`. To target only the .rlat:

```bash
kaggle kernels output <username>/<slug> -p ./outputs/ \
   --file-pattern "<name>.rlat"
```

**Large file download quirk (>~300 MB)**: the CLI silently buffers the
whole file in memory before writing to disk. Output appears stuck at 0
bytes for 3-5 minutes, then writes in one burst. Use a long timeout (15+
min) and **don't kill the command** if the file size hasn't moved — poll
file size as the liveness signal, not stdout.

## Step 7 — Verify the .rlat locally

```bash
# Magic bytes: real ZIP archive starts with PK\x03\x04
python -c "print(open('outputs/my-docs.rlat', 'rb').read(4))"
# Expected: b'PK\x03\x04'

# Open it and inspect
rlat profile outputs/my-docs.rlat

# First query (downloads source from raw.githubusercontent.com on demand)
rlat search outputs/my-docs.rlat "your test query" --top-k 3
```

If `profile` reports `passage_count`, `band` info, and the build's
`commit_sha`, the build worked. The first remote-mode query is slow
(network fetches per hit); subsequent queries against the same passages
are fast (the source bytes are cached locally).

## Multi-corpus batches (resilient pattern)

If you're rebuilding several `.rlat` files in one Kaggle session,
structure the script around a checkpoint file in `/kaggle/working/` so a
late failure doesn't lose earlier successes:

```python
RESULTS = WORK / "build_results.json"
results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}

for corpus in CORPORA:
    name = corpus["name"]
    out = WORK / f"{name}.rlat"
    # Skip if result-record AND output file are present
    if results.get(name, {}).get("status") == "ok" and out.exists():
        print(f"[{name}] already done, skipping")
        continue
    try:
        results[name] = build_one(corpus)
    except Exception as e:
        results[name] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        traceback.print_exc()
    # Persist after EVERY corpus — this is the resilience anchor
    RESULTS.write_text(json.dumps(results, indent=2))
```

The in-kernel filesystem persists between iterations of one run. Each
`.rlat` that hits `/kaggle/working/` will ship in the output bucket
even if a later corpus crashes.

For genuinely cross-push resume (e.g. a partial run on Monday + finish
on Tuesday), attach the kernel's previous outputs as a `dataset_sources`
entry — Kaggle's "version this kernel" doesn't preserve `/kaggle/working/`
across versions.

## Common gotchas (in failure-frequency order)

| Symptom | Cause | Fix |
|---|---|---|
| `'charmap' codec can't decode byte ...` traceback on push | Windows CP1252 decode of server response | `setx PYTHONUTF8 1` once; new terminal; re-push. The server-side push almost certainly succeeded — re-push is idempotent. |
| `409 Conflict for url: …KernelsApiService/SaveKernel` on re-push | `id` ≠ `slugify(title)` after first push registered the title slug | Make `id` and `slugify(title)` match (lowercase + hyphens), then re-push |
| `status` returns 403 right after push | Charmap crash silently ate the success record | `setx PYTHONUTF8 1`, re-push |
| Encode finishes, `.rlat` build crashes with `RuntimeError: faiss is not installed` | `pip install rlat[build]` doesn't include `[ann]`; corpora >5000 passages need it | Use `pip install rlat[build,ann]` |
| Kernel runs "forever" then runs out of compute | P100 assigned (sm_60); CUDA falls back to CPU silently | Always pass `--accelerator NvidiaTeslaT4` on push |
| Local `kaggle kernels output` says "stuck at 0 bytes" but no error | Large file (>300 MB) — CLI buffers before writing | Wait 3-5 min; do not kill |
| Mid-run web UI shows "Output 0 B" | Kaggle ships outputs only at terminal state | Normal — check the log panel for live progress instead |
| `ModuleNotFoundError: torch` after install completes | rlat[build] base install conflict, or pip didn't pick up the extras | `pip install --upgrade --force-reinstall rlat[build,ann]` |
| Stderr lines appear before stdout from earlier timestamps | Stdout/stderr flush at different times in Kaggle log | Trust the timestamp column, not visual order |

## Resource limits to budget against

| Limit | Value |
|---|---|
| GPU quota | 30 hours / week (free tier) |
| Max execution time | 12 hours / kernel session |
| Disk in `/kaggle/working/` | ~73 GB |
| Output size shipped to user | 20 GB max |
| RAM (T4) | ~13 GB |

A single `gte-modernbert-base` encode at batch 64 uses ~3 GB GPU memory.
A 100K-passage corpus encodes in ~2-5 minutes. The bottleneck for big
batches is usually the source clone (network) rather than the encode.

## Wrapping up

After the run:

1. The `.rlat` is in `outputs/<name>.rlat`. Move it wherever your
   project expects (`./<name>.rlat`, `data/<name>.rlat`, etc.).
2. If you used remote mode, make sure the consumer machine has internet
   access at query time — passages are streamed from
   `raw.githubusercontent.com`. Cache lives at
   `~/.cache/rlat/sources/<sha>/` and warms on first use.
3. Smoke-test with `rlat search <name>.rlat "anything"` locally before
   shipping the artefact further.

For deeper background on what `rlat build` does, see
[docs/user/CLI.md](../../../docs/user/CLI.md) (the `build` section).
For the storage-mode trade-offs (why remote? why bundled?), see
[docs/user/STORAGE_MODES.md](../../../docs/user/STORAGE_MODES.md).
