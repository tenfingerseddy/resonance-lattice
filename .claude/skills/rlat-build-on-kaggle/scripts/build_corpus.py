"""Build a single rlat knowledge model on a Kaggle T4 (remote storage mode).

This is the single-corpus template. For multi-corpus batches with
per-iteration checkpointing, see build_corpora_batch.py.

Adapt the CONFIG block below to your corpus, then push the kernel:

    PYTHONUTF8=1 kaggle kernels push -p kaggle/<your-dir> --accelerator NvidiaTeslaT4

The kernel writes /kaggle/working/<name>.rlat — this file ships out at
COMPLETE / ERROR. Pull it home with:

    PYTHONUTF8=1 kaggle kernels output <username>/<slug> -p ./outputs/ -o
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
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


def shell(cmd: list[str], log: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command, tee stdout+stderr to log if provided. Raises on non-zero."""
    print(f"$ {' '.join(cmd)}", flush=True)
    if log is None:
        return subprocess.run(cmd, check=True)
    with open(log, "ab") as f:
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


def main() -> int:
    # 1. Install rlat with the GPU build extras AND the ANN extras.
    #    `[ann]` pulls faiss-cpu — required for FAISS HNSW above ~5000
    #    passages. Forgetting it lets the encode finish then crashes
    #    with "RuntimeError: faiss is not installed".
    shell([sys.executable, "-m", "pip", "install", "--quiet", "rlat[build,ann]"])
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
        ["git", "-C", str(target), "rev-parse", "FETCH_HEAD"], text=True
    ).strip()
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

    # 5. Result summary so build_results.json reports success cleanly.
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
