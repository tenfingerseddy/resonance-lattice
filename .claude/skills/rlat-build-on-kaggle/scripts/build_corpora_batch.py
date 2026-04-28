"""Build several rlat knowledge models on a Kaggle T4 in one session.

Resilient batch pattern: each completed `.rlat` is written to
/kaggle/working/ as soon as it's done — if a later corpus fails, earlier
successes are preserved in the kernel's COMPLETE output. Per-corpus
errors are caught and logged, never fatal to the batch. Final exit code
is 0 if all succeeded, 1 otherwise (so the kernel surfaces failure but
still ships outputs).

Adapt the CORPORA list, then push:

    PYTHONUTF8=1 kaggle kernels push -p kaggle/<your-dir> --accelerator NvidiaTeslaT4

Outputs:
  /kaggle/working/<name>.rlat                # the rebuilt KMs
  /kaggle/working/build_results.json         # per-corpus status (rewritten after every corpus)
  /kaggle/working/<name>.build.log           # per-corpus stdout+stderr
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ── Config: list every corpus you want to rebuild ───────────────────
CORPORA = [
    {
        "name": "my-docs",
        "github": "owner/repo",
        "branch": "main",
        "scope": "docs",
    },
    {
        "name": "my-other-docs",
        "github": "owner/other-repo",
        "branch": "main",
        "scope": "reference",
    },
    # add more …
]
# ────────────────────────────────────────────────────────────────────

WORK = Path("/kaggle/working")
WORK.mkdir(exist_ok=True, parents=True)
SRC_ROOT = WORK / "_src"
RESULTS_PATH = WORK / "build_results.json"


def shell(cmd: list[str], log_path: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command, tee stdout+stderr to log_path if provided. Raises on non-zero."""
    print(f"$ {' '.join(cmd)}", flush=True)
    if log_path is None:
        return subprocess.run(cmd, check=True)
    with open(log_path, "ab") as f:
        f.write(f"\n$ {' '.join(cmd)}\n".encode())
        f.flush()
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
        )
        f.write(proc.stdout)
    sys.stdout.write(proc.stdout.decode("utf-8", errors="replace"))
    sys.stdout.flush()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc


def install_rlat() -> None:
    """One-time rlat install + encoder download + sanity-print."""
    print("=== install rlat[build,ann] + encoder ===", flush=True)
    # [build]: transformers/torch/onnxscript for first build + ONNX export.
    # [ann]:   faiss-cpu — required for ANN HNSW index build at N >= 5000
    #          passages. Without it, build aborts with a clear error
    #          AFTER encoding finishes — wasting all the GPU time.
    shell([sys.executable, "-m", "pip", "install", "--quiet", "rlat[build,ann]"])
    shell(["rlat", "install-encoder"])
    shell([sys.executable, "-c",
           "import resonance_lattice as rl; print('rlat', rl.__version__);"
           "from resonance_lattice.install.encoder import MODEL_ID, PINNED_REVISION;"
           "print('encoder', MODEL_ID, '@', PINNED_REVISION);"
           "import torch; print('cuda available:', torch.cuda.is_available(),"
           " '| device count:', torch.cuda.device_count());"
           "print('cuda device 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"])


def sparse_clone(github: str, branch: str, scope: str, target: Path) -> str:
    """Sparse-clone a single subdir of a GitHub repo. Returns commit SHA."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    shell(["git", "init", "-q", "-b", branch, str(target)])
    shell(["git", "-C", str(target), "remote", "add", "origin",
           f"https://github.com/{github}.git"])
    shell(["git", "-C", str(target), "config", "core.sparseCheckout", "true"])
    (target / ".git/info/sparse-checkout").write_text(f"{scope}/*\n")
    shell(["git", "-C", str(target), "fetch", "--depth", "1", "origin", branch])
    shell(["git", "-C", str(target), "checkout", "FETCH_HEAD"])
    sha = subprocess.check_output(
        ["git", "-C", str(target), "rev-parse", "FETCH_HEAD"], text=True
    ).strip()
    return sha


def build_one(corpus: dict, log_path: Path) -> dict:
    """Build one corpus into /kaggle/working/<name>.rlat. Returns result dict."""
    name = corpus["name"]
    out = WORK / f"{name}.rlat"
    t0 = time.time()

    if out.exists():
        return {
            "status": "ok",
            "skipped": True,
            "rlat_path": str(out),
            "size_bytes": out.stat().st_size,
            "wall_seconds": 0,
        }

    src_dir = SRC_ROOT / corpus["github"].replace("/", "_")
    sha = sparse_clone(corpus["github"], corpus["branch"], corpus["scope"], src_dir)
    print(f"[{name}] pinned to {corpus['github']}@{sha}", flush=True)

    scope_dir = src_dir / corpus["scope"] if corpus["scope"] else src_dir
    if not scope_dir.exists():
        raise RuntimeError(
            f"{name}: scope path '{corpus['scope']}' not found under {src_dir}. "
            f"Repo layout may differ from expectation. Listing top-level:\n"
            + "\n".join(sorted(p.name for p in src_dir.iterdir() if not p.name.startswith(".")))
        )

    url_base = f"https://raw.githubusercontent.com/{corpus['github']}/{sha}/{corpus['scope']}/"

    print(f"[{name}] start build (runtime=torch, batch_size=64) at {time.strftime('%H:%M:%S')}",
          flush=True)
    shell([
        "rlat", "build",
        str(scope_dir),
        "-o", str(out),
        "--store-mode", "remote",
        "--remote-url-base", url_base,
        "--runtime", "torch",
        "--batch-size", "64",
    ], log_path=log_path)

    if not out.exists():
        raise RuntimeError(f"{name}: rlat build returned 0 but {out} not on disk")

    # Free disk: drop the cloned source after a successful build
    shutil.rmtree(src_dir, ignore_errors=True)

    result = {
        "status": "ok",
        "rlat_path": str(out),
        "size_bytes": out.stat().st_size,
        "github": corpus["github"],
        "scope": corpus["scope"],
        "commit_sha": sha,
        "remote_url_base": url_base,
        "wall_seconds": round(time.time() - t0, 1),
    }
    print(f"[{name}] DONE  size={result['size_bytes']:,} bytes  wall={result['wall_seconds']}s",
          flush=True)
    return result


def main() -> int:
    print(f"=== rlat batch corpus build — {len(CORPORA)} corpora ===", flush=True)

    # Resume support: load any prior results so a re-run skips completed corpora.
    if RESULTS_PATH.exists():
        try:
            results = json.loads(RESULTS_PATH.read_text())
            print(f"resuming from existing build_results.json: {list(results.keys())}", flush=True)
        except Exception:
            results = {}
    else:
        results = {}

    install_rlat()

    SRC_ROOT.mkdir(exist_ok=True, parents=True)

    for corpus in CORPORA:
        name = corpus["name"]
        log_path = WORK / f"{name}.build.log"
        # Skip if already ok in results AND output file present
        prior = results.get(name)
        if prior and prior.get("status") == "ok" and (WORK / f"{name}.rlat").exists():
            print(f"\n=== [{name}] already complete, skipping ===", flush=True)
            continue

        print(f"\n=== [{name}] ===", flush=True)
        t0 = time.time()
        try:
            results[name] = build_one(corpus, log_path)
        except Exception as e:
            results[name] = {
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
                "wall_seconds": round(time.time() - t0, 1),
            }
            print(f"[{name}] FAILED: {e}", flush=True)
            traceback.print_exc()
        # Persist results-so-far after every corpus (resilience checkpoint)
        RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== summary ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)

    failed = [n for n, r in results.items() if r.get("status") != "ok"]
    if failed:
        print(f"\nFAILED: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
