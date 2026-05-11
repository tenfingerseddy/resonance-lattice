"""fabric_client — `rlat search fabric://...` URL dispatch contract +
`rlat fabric add/list/remove` subcommand contract.

Eight guarantees against a fake HTTP server (no real Fabric / Entra):

  1. fabric://<alias>/<km> POSTs to <endpoint>/search with the right body
     and renders the response as text/json/context.
  2. fabric://<alias> (no km) POSTs to <endpoint>/list_kms and prints the
     discovery rows as JSON.
  3. Unregistered alias exits cleanly with a "rlat fabric add" hint.
  4. URL parser splits alias and km correctly, accepts trailing slashes.
  5. HTTPError on the UDF endpoint surfaces as a non-zero exit with the
     server's response body in stderr.
  6. `rlat fabric add team=<url>` writes the TOML, scaffolds the skill,
     and a follow-up `rlat search fabric://team` finds the alias.
  7. `rlat fabric list` prints registered aliases (one line each).
  8. `rlat fabric remove team` deletes the alias and updates the skill
     scaffold's frontmatter description.
  9. _fabric_auth._have_sp_env() returns True iff all three AZURE_*
     env vars are set (otherwise device-code is the default path).
 10. _fabric_auth.get_token() raises FabricAuthError with a "[fabric] extra"
     hint when azure.identity is unavailable.

Auth (`_get_token`) is monkeypatched to return a fake bearer so the
suite doesn't need azure-identity or a live Entra tenant in guarantees
1-8; guarantees 9-10 exercise `_fabric_auth` directly.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from ._testutil import Args, check_guarantee, run_cli  # noqa: F401  (Args reserved for future cases)


# ---------- fake UDF server ------------------------------------------------

_FAKE_HITS = [
    {
        "passage_idx": 1, "source_file": "deploy.md",
        "char_offset": 0, "char_length": 32, "content_hash": "abc",
        "drift_status": "verified", "score": 0.81,
        "text": "Deploy via the Fabric pipeline.",
    },
    {
        "passage_idx": 2, "source_file": "auth.md",
        "char_offset": 0, "char_length": 24, "content_hash": "def",
        "drift_status": "verified", "score": 0.62,
        "text": "Auth uses Entra device code.",
    },
]
_FAKE_KMS = [
    {"kmName": "team-docs", "n_passages": 4823,
     "created_utc": "2026-05-01T12:00:00Z",
     "encoder_revision": "e7f32e3c00f91d699e8c43b53106206bcc72bb22"},
]


class _UDFHandler(BaseHTTPRequestHandler):
    captured: list[dict] = []  # class-level so the test can assert post-hoc

    def log_message(self, format, *args):  # noqa: ARG002
        return  # silence default stderr access logs

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        path = self.path.rstrip("/")
        self.captured.append({"path": path, "body": body,
                              "auth": self.headers.get("Authorization", "")})

        # Mirrors the real Fabric UDF wire format:
        #   URL: <base>/functions/<name>/invoke
        #   resp: {"functionName": ..., "status": "Succeeded", "output": <return>, "errors": []}

        # `/error/...` always 500s. Checked first so `/error/functions/.../invoke`
        # doesn't fall through to the OK branch.
        if "/error/" in path or path.endswith("/error"):
            self._respond(500, {"error": "boom"})
            return
        if not path.endswith("/invoke"):
            self._respond(404, {"error": f"unknown path {path}"})
            return

        if path.endswith("/functions/search/invoke"):
            self._envelope("search", {"band": "base", "cold": False, "hits": _FAKE_HITS})
        elif path.endswith("/functions/list_kms/invoke"):
            self._envelope("list_kms", _FAKE_KMS)
        else:
            self._respond(404, {"error": f"unknown function path {path}"})

    def _envelope(self, function_name: str, output) -> None:
        self._respond(200, {
            "functionName": function_name,
            "status":       "Succeeded",
            "output":       output,
            "errors":       [],
        })

    def _respond(self, code: int, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _spawn_server():
    port = _free_port()
    srv = HTTPServer(("127.0.0.1", port), _UDFHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", _UDFHandler
    finally:
        srv.shutdown()
        thread.join(timeout=1.0)


# ---------- patches --------------------------------------------------------

def _write_alias_config(tmp_root: Path, alias: str, url: str) -> Path:
    cfg = tmp_root / "fabric.toml"
    cfg.write_text(f'[aliases.{alias}]\nurl = "{url}"\n', encoding="utf-8")
    return cfg


def _patch_token():
    import resonance_lattice.cli._fabric as fab
    fab._get_token = lambda: "fake-bearer"  # type: ignore[assignment]


# ---------- guarantees -----------------------------------------------------

def _check(ok: bool, label: str) -> bool:
    return check_guarantee(ok, label, "fabric_client")


def run() -> int:
    import tempfile
    failures = 0

    _patch_token()
    _UDFHandler.captured.clear()

    with tempfile.TemporaryDirectory() as d, _spawn_server() as (endpoint, handler):
        cfg = _write_alias_config(Path(d), "team", endpoint)
        os.environ["RLAT_FABRIC_CONFIG"] = str(cfg)

        # ---- Guarantee 1: fabric://team/docs runs search ----
        rc, out, err = run_cli([
            "search", "fabric://team/team-docs", "how do I deploy?",
            "--top-k", "5", "--quiet",
        ])
        last = handler.captured[-1] if handler.captured else {}
        passed_g1 = (
            rc == 0
            and last.get("path", "").endswith("/functions/search/invoke")
            and last["body"]["kmName"] == "team-docs"
            and last["body"]["query"] == "how do I deploy?"
            and last["body"]["topK"] == 5
            and last["auth"] == "Bearer fake-bearer"
            and "Deploy via the Fabric pipeline" in out
        )
        failures += not _check(passed_g1, "guarantee 1 (fabric:// search)")

        # ---- Guarantee 2: fabric://team (no km) runs list_kms ----
        rc, out, err = run_cli([
            "search", "fabric://team", "ignored-query", "--quiet",
        ])
        last = handler.captured[-1] if handler.captured else {}
        passed_g2 = (
            rc == 0
            and last.get("path", "").endswith("/functions/list_kms/invoke")
            and "team-docs" in out
            and "n_passages" in out
        )
        failures += not _check(passed_g2, "guarantee 2 (no-km -> list_kms)")

        # ---- Guarantee 3: unregistered alias exits with hint ----
        rc, out, err = run_cli([
            "search", "fabric://no-such-alias/km", "q", "--quiet",
        ])
        passed_g3 = rc == 2 and "rlat fabric add" in err
        failures += not _check(passed_g3, "guarantee 3 (unknown alias)")

        # ---- Guarantee 4: URL parser handles trailing slashes ----
        from resonance_lattice.cli._fabric import _parse_fabric_url
        passed_g4 = (
            _parse_fabric_url("fabric://team") == ("team", None)
            and _parse_fabric_url("fabric://team/") == ("team", None)
            and _parse_fabric_url("fabric://team/docs") == ("team", "docs")
            and _parse_fabric_url("fabric://team/docs/") == ("team", "docs")
        )
        failures += not _check(passed_g4, "guarantee 4 (URL parser)")

        # ---- Guarantee 5: HTTP 500 surfaces as exit 2 + stderr ----
        # Point the alias at /error to force a 500. urllib raises HTTPError
        # which _post_udf catches and prints to stderr.
        bad_cfg = _write_alias_config(Path(d), "broken", endpoint + "/error")
        os.environ["RLAT_FABRIC_CONFIG"] = str(bad_cfg)
        # Re-write so _load_aliases re-reads with the new alias.
        Path(bad_cfg).write_text(
            f'[aliases.team]\nurl = "{endpoint}"\n'
            f'[aliases.broken]\nurl = "{endpoint}/error"\n',
            encoding="utf-8",
        )
        rc, out, err = run_cli([
            "search", "fabric://broken/anything", "q", "--quiet",
        ])
        passed_g5 = rc == 2 and "500" in err and "boom" in err
        failures += not _check(passed_g5, "guarantee 5 (HTTPError -> exit 2)")

        # Restore the team-only config for the subcommand guarantees.
        cfg2 = _write_alias_config(Path(d), "team", endpoint)
        os.environ["RLAT_FABRIC_CONFIG"] = str(cfg2)

        # ---- Guarantee 6: `rlat fabric add` writes config + scaffolds skill ----
        sub_d = Path(d) / "subcmd"
        (sub_d).mkdir(parents=True, exist_ok=True)
        # Use a clean config so `add` writes from an empty state.
        cfg3 = sub_d / "fabric.toml"
        os.environ["RLAT_FABRIC_CONFIG"] = str(cfg3)
        rc, out, err = run_cli([
            "fabric", "--cwd", str(sub_d),
            "add", f"team={endpoint}",
        ])
        skill_md = sub_d / ".claude/skills/rlat-fabric-search/SKILL.md"
        passed_g6 = (
            rc == 0
            and cfg3.exists()
            and f'url = "{endpoint}"' in cfg3.read_text(encoding="utf-8")
            and skill_md.exists()
            and "team" in skill_md.read_text(encoding="utf-8")
        )
        # Round-trip: a follow-up search should now resolve the alias.
        rc2, out2, err2 = run_cli([
            "search", "fabric://team", "q", "--quiet",
        ])
        passed_g6 = passed_g6 and rc2 == 0 and "team-docs" in out2
        failures += not _check(passed_g6, "guarantee 6 (rlat fabric add)")

        # ---- Guarantee 7: `rlat fabric list` prints aliases ----
        rc, out, err = run_cli(["fabric", "list"])
        passed_g7 = rc == 0 and "team" in out and endpoint in out
        failures += not _check(passed_g7, "guarantee 7 (rlat fabric list)")

        # ---- Guarantee 8: `rlat fabric remove` drops alias + rewrites skill ----
        rc, out, err = run_cli([
            "fabric", "--cwd", str(sub_d),
            "remove", "team",
        ])
        skill_after = skill_md.read_text(encoding="utf-8")
        passed_g8 = (
            rc == 0
            and "team" not in cfg3.read_text(encoding="utf-8")
            and "no aliases registered yet" in skill_after
        )
        failures += not _check(passed_g8, "guarantee 8 (rlat fabric remove)")

        # ---- Guarantee 9: SP env-var detection ----
        from resonance_lattice.cli import _fabric_auth as fa

        prior = {k: os.environ.get(k) for k in
                 ("AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID")}
        try:
            for k in prior:
                os.environ.pop(k, None)
            no_sp = fa._have_sp_env()
            for k in prior:
                os.environ[k] = "x"
            yes_sp = fa._have_sp_env()
            passed_g9 = (no_sp is False) and (yes_sp is True)
        finally:
            for k, v in prior.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        failures += not _check(passed_g9, "guarantee 9 (SP env detection)")

        # ---- Guarantee 10: missing azure.identity -> FabricAuthError ----
        # Block the import by injecting a dummy entry; restore after.
        prior_identity = sys.modules.pop("azure.identity", None)
        sys.modules["azure.identity"] = None  # type: ignore[assignment]
        try:
            try:
                fa._make_credential()
            except fa.FabricAuthError as e:
                passed_g10 = "[fabric] extra" in str(e)
            else:
                passed_g10 = False
        finally:
            sys.modules.pop("azure.identity", None)
            if prior_identity is not None:
                sys.modules["azure.identity"] = prior_identity
        failures += not _check(passed_g10, "guarantee 10 (missing azure.identity)")

    if failures:
        print(f"[fabric_client] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[fabric_client] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
