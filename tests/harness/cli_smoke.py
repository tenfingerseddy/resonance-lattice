"""cli_smoke — every CLI command parses, and init-project survives end-to-end.

The 2026-06 review found `rlat init-project` crashing on a hand-built
Namespace whose attribute name drifted from its consumer (`queries` vs
`args.query`) — a rename miss no suite caught because nothing dispatches
the commands. These guarantees close that class:

  K1. Every subcommand registered on the real parser handles `--help`
      (argparse SystemExit(0)) — catches registration/import breakage
      across the whole surface in one cheap pass.
  K2. `init-project` runs end-to-end (build → summary primer) in a temp
      project with the stub encoder: rc == 0 and the primer file exists.
      Exercises the hand-built Namespaces against their real consumers —
      the regression test for the `queries`→`query` fix.
  K3. `init-project --no-primer` short-circuits with rc == 0 and writes
      no primer.
  K4. `grow --dry-run` on a fresh archive (no telemetry) is a clean
      no-op — pins the opt-in self-improvement surface's wiring.
  K6. pyproject `[project] version` == `resonance_lattice.__version__`
      (release parity; pyproject is the source of truth).
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

from ._testutil import check_guarantee, patch_zero_encoder, unpatch_zero_encoder


def _check(ok: bool, label: str) -> bool:
    return check_guarantee(ok, label, "cli_smoke")


def _subcommands(parser) -> dict[str, object]:
    """The registered subcommand map off the real parser (test-only probe)."""
    for action in parser._actions:  # noqa: SLF001
        if hasattr(action, "choices") and action.choices:
            return dict(action.choices)
    return {}


def _run_init(tmp: Path, extra: list[str]) -> int:
    """Dispatch init-project through the real parse → func path from `tmp`.

    Any escape — including the `sys.exit(1)` paths in cli/_load.py and
    argparse's own SystemExit — becomes a failed guarantee, never a crashed
    suite, so the rest of a multi-suite sweep still reports.
    """
    from resonance_lattice.cli.app import build_parser

    prev = Path.cwd()
    os.chdir(tmp)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            args = build_parser().parse_args(["init-project", *extra])
            return int(args.func(args))
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
        print(f"[cli_smoke] init-project SystemExit({rc}); output:\n{out.getvalue()}",
              file=sys.stderr)
        return rc if rc != 0 else -1
    except Exception as exc:  # crash == failed guarantee, not a crashed suite
        print(f"[cli_smoke] init-project raised {type(exc).__name__}: {exc}; output:\n{out.getvalue()}",
              file=sys.stderr)
        return -1
    finally:
        os.chdir(prev)


def _check_install_hooks() -> int:
    """K5. `memory install-hooks` is an idempotent merge: two runs yield
    one rlat entry per event; a pre-existing foreign hook and env key
    survive untouched; --mine sets the mining opt-in exactly once."""
    import json

    from resonance_lattice.cli.app import build_parser
    from resonance_lattice.cli.memory import merge_hook_settings

    ok = True
    with tempfile.TemporaryDirectory() as td:
        proj = Path(td)
        cfg = proj / ".claude" / "settings.json"
        cfg.parent.mkdir(parents=True)
        foreign = {"matcher": "*", "hooks": [
            {"type": "command", "command": "echo keepme"}]}
        cfg.write_text(json.dumps(
            {"hooks": {"SessionEnd": [foreign]}, "env": {"FOO": "bar"}}),
            encoding="utf-8")
        args = build_parser().parse_args(
            ["memory", "install-hooks", "--project-dir", str(proj), "--mine"])
        with contextlib.redirect_stdout(io.StringIO()):
            rc1 = int(args.func(args))
            rc2 = int(args.func(args))          # second run: no changes
        data = json.loads(cfg.read_text(encoding="utf-8"))
        se = data["hooks"]["SessionEnd"]
        rlat_se = [e for e in se
                   if any(h.get("command") == "rlat memory capture"
                          for h in e.get("hooks", []))]
        ups = data["hooks"].get("UserPromptSubmit", [])
        rlat_up = [e for e in ups
                   if any(h.get("command") == "rlat memory hook"
                          for h in e.get("hooks", []))]
        ok &= _check(rc1 == 0 and rc2 == 0, f"K5: both runs rc==0 ({rc1},{rc2})")
        ok &= _check(len(rlat_se) == 1 and len(rlat_up) == 1,
                     "K5: exactly one rlat entry per event after two runs")
        ok &= _check(foreign in se, "K5: foreign SessionEnd hook preserved")
        ok &= _check(data["env"].get("FOO") == "bar"
                     and data["env"].get("RLAT_MINE_ATTRIBUTES") == "1",
                     "K5: env preserved + mining opt-in set")

        # Malformed-but-valid-JSON shapes fail gracefully, file untouched.
        cfg.write_text(json.dumps({"hooks": []}), encoding="utf-8")
        before = cfg.read_text(encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                rc3 = int(args.func(args))
            except Exception as exc:
                print(f"[cli_smoke] K5 raised {type(exc).__name__}", file=sys.stderr)
                rc3 = -1
        ok &= _check(rc3 == 1, f"K5: malformed hooks shape -> rc==1, no crash (got {rc3})")
        ok &= _check(cfg.read_text(encoding="utf-8") == before,
                     "K5: malformed file left untouched")
    return 0 if ok else 1


def run() -> int:
    patch_zero_encoder()
    try:
        return _run_all()
    finally:
        unpatch_zero_encoder()


def _run_all() -> int:
    from resonance_lattice.cli.app import build_parser

    ok = True
    parser = build_parser()
    commands = _subcommands(parser)
    ok &= _check(len(commands) >= 20, f"K1a: subcommand registry populated ({len(commands)} found)")

    for name in sorted(commands):
        code: int | None = None
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                build_parser().parse_args([name, "--help"])
            except SystemExit as exc:  # argparse exits 0 on --help
                code = int(exc.code or 0)
        ok &= _check(code == 0, f"K1: `rlat {name} --help` parses (exit {code})")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td).resolve()
        (tmp / "docs").mkdir()
        (tmp / "docs" / "guide.md").write_text(
            "# Guide\n\n" + ("Resonance lattice smoke corpus paragraph. " * 12),
            encoding="utf-8",
        )
        (tmp / "README.md").write_text(
            "# Smoke\n\n" + ("A readme long enough to chunk into a passage. " * 12),
            encoding="utf-8",
        )

        rc = _run_init(tmp, [])
        primer = tmp / ".claude" / "resonance-context.md"
        ok &= _check(rc == 0, f"K2: init-project end-to-end rc==0 (got {rc})")
        ok &= _check(primer.is_file() and primer.stat().st_size > 0,
                     "K2: primer written to .claude/resonance-context.md")

        # K4: `rlat grow --dry-run` on the freshly-built archive — an
        # empty-telemetry corpus is a clean no-op (rc 0, no LLM, no writes).
        km = next(tmp.glob("*.rlat"), None)
        if km is not None:
            try:
                grow_args = build_parser().parse_args(
                    ["grow", str(km), "--dry-run"])
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    grow_rc = int(grow_args.func(grow_args))
            except (Exception, SystemExit) as exc:
                print(f"[cli_smoke] grow --dry-run raised "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                grow_rc = -1
            ok &= _check(grow_rc == 0,
                         f"K4: grow --dry-run no-telemetry no-op (rc {grow_rc})")
        else:
            ok &= _check(False, "K4: no .rlat found after init-project")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td).resolve()
        (tmp / "README.md").write_text(
            "# Smoke\n\n" + ("No-primer branch corpus text for chunking. " * 12),
            encoding="utf-8",
        )
        rc = _run_init(tmp, ["--no-primer"])
        ok &= _check(rc == 0, f"K3: --no-primer rc==0 (got {rc})")
        ok &= _check(not (tmp / ".claude" / "resonance-context.md").exists(),
                     "K3: no primer written")

    ok &= _check_install_hooks() == 0
    ok &= _check_version_parity() == 0

    return 0 if ok else 1


def _check_version_parity() -> int:
    """K6. `pyproject.toml` `[project] version` equals
    `resonance_lattice.__version__` — the two hardcoded version strings can
    never drift (release-engineering guarantee; pyproject is the source of
    truth)."""
    import re

    import resonance_lattice

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"),
                  re.MULTILINE)
    declared = m.group(1) if m else "<missing>"
    ok = _check(declared == resonance_lattice.__version__,
                f"K6: version parity (pyproject {declared} == "
                f"__version__ {resonance_lattice.__version__})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
