"""SessionEnd hook — ingest the closing transcript into layered memory.

On Claude Code session close, this hook reads the session transcript and
calls ``rlat memory write`` to append it to the project's layered memory
store (``working`` tier). First use auto-initializes the memory root.

Env var controls (all optional):

  - ``RLAT_MEMORY_ROOT``    — override memory root (default: ``<cwd>/memory``)
  - ``RLAT_OPENVINO_DIR``   — OpenVINO IR dir for fast encode (Intel Arc)
  - ``RLAT_OPENVINO_DEVICE``— ``CPU`` | ``GPU`` | ``NPU`` | ``AUTO``
  - ``RLAT_ONNX_DIR``       — ONNX backbone dir (alternative accelerator)
  - ``RLAT_REBUILD_PRIMER`` — ``1`` to regenerate memory primer after write
  - ``RLAT_CODE_CARTRIDGE`` — code cartridge path for primer regen

Failure policy: every external call is guarded; exceptions are written to
stderr and the hook exits 0 so session close is never blocked.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _hookutil as H  # noqa: E402


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
            timeout=300,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except Exception as exc:
        return -1, "", f"{type(exc).__name__}: {exc}"


def main() -> None:
    with H.guard("memory-session-end"):
        payload = H.read_hook_input()
        sid = H.session_id_from_input(payload)
        transcript_path = payload.get("transcript_path") or ""
        cwd = H.project_dir(payload)

        if not transcript_path or not Path(transcript_path).is_file():
            return

        memory_root = Path(os.environ.get("RLAT_MEMORY_ROOT") or cwd / "memory")
        if not memory_root.exists():
            rc, _out, err = _run(["rlat", "memory", "init", str(memory_root)])
            if rc != 0:
                sys.stderr.write(f"[rlat-hook] memory init failed rc={rc}: {err[:400]}\n")
                return

        cmd = [
            "rlat", "memory", "write", str(memory_root),
            "--input-file", transcript_path,
            "--input-format", "claude_transcript",
            "--session", sid,
            "--tier", "working",
        ]
        ov_dir = os.environ.get("RLAT_OPENVINO_DIR")
        if ov_dir and Path(ov_dir).is_dir():
            cmd += ["--openvino", ov_dir]
            ov_dev = os.environ.get("RLAT_OPENVINO_DEVICE")
            if ov_dev:
                cmd += ["--openvino-device", ov_dev]
        onnx_dir = os.environ.get("RLAT_ONNX_DIR")
        if not ov_dir and onnx_dir and Path(onnx_dir).is_dir():
            cmd += ["--onnx", onnx_dir]

        rc, _out, err = _run(cmd)
        if rc != 0:
            sys.stderr.write(
                f"[rlat-hook] memory write failed rc={rc}: {err[:500]}\n"
                f"[rlat-hook] cmd={shlex.join(cmd)}\n"
            )
            return

        if os.environ.get("RLAT_REBUILD_PRIMER") == "1":
            primer_cmd = ["rlat", "memory", "primer", str(memory_root)]
            code_cart = os.environ.get("RLAT_CODE_CARTRIDGE")
            if code_cart and Path(code_cart).is_file():
                primer_cmd += ["--code-cartridge", code_cart]
            rc, _out, err = _run(primer_cmd)
            if rc != 0:
                sys.stderr.write(f"[rlat-hook] memory primer failed rc={rc}: {err[:400]}\n")


if __name__ == "__main__":
    main()
