"""Client-side `fabric://` URL dispatch for `rlat search`.

URL forms:
  fabric://<alias>          → call list_kms() endpoint, print the table
  fabric://<alias>/<km>     → call search() endpoint with kmName=<km>

Aliases are resolved against `~/.config/rlat/fabric.toml`, written by
`rlat fabric add <alias>=<url>`. The TOML schema is:

    [aliases.<alias>]
    url = "<UDF base URL ending in /userDataFunctions/<udf-id>>"

For each invocation, the client appends `/functions/<name>/invoke` to
get the per-function URL. The Fabric UDF wire shape:

    POST <endpoint>/functions/<name>/invoke
    body: {<param>: <value>, ...}                   ← params at top level
    resp: {functionName, status, output, errors}    ← return is in `output`
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from ..store.verified import VerifiedHit
from .search import _format_context, _format_json, _format_text
from . import _grounding, _namecheck

CONFIG_PATH_ENV = "RLAT_FABRIC_CONFIG"
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "rlat" / "fabric.toml"


def _config_path() -> Path:
    """Resolve the fabric.toml path. Test override via env var."""
    import os
    override = os.environ.get(CONFIG_PATH_ENV)
    return Path(override) if override else DEFAULT_CONFIG_PATH


def _load_aliases() -> dict[str, dict]:
    """Return the `[aliases.*]` table from fabric.toml, or {} if absent."""
    import tomllib
    path = _config_path()
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f).get("aliases", {})


def _resolve_alias(alias: str) -> str | None:
    """Return the UDF endpoint URL for `alias`, or None if unregistered."""
    entry = _load_aliases().get(alias)
    if not entry or "url" not in entry:
        return None
    return entry["url"].rstrip("/")


def _parse_fabric_url(url: str) -> tuple[str, str | None]:
    """Split `fabric://<alias>[/<km>]` into `(alias, km_or_None)`."""
    if not url.startswith("fabric://"):
        raise ValueError(f"not a fabric URL: {url!r}")
    body = url.removeprefix("fabric://").rstrip("/")
    if "/" in body:
        alias, km = body.split("/", 1)
        return alias, km or None
    return body, None


def _get_token() -> str:
    """Tests monkeypatch this symbol directly; production calls into
    `_fabric_auth.get_token()` (SP env vars when set, device-code otherwise)."""
    from ._fabric_auth import get_token
    return get_token()


class _UDFCallFailed(RuntimeError):
    """Raised by `_post_udf` on HTTP error so the caller can return an rc."""


def _post_udf(endpoint: str, function_name: str, parameters: dict) -> dict | list:
    """Invoke a UDF function and return its `output` (already unwrapped from
    the {functionName, status, output, errors} envelope).

    Raises `_UDFCallFailed` on HTTP error or `status != "Succeeded"`.
    """
    body = json.dumps(parameters).encode()
    headers = {
        "Authorization": f"Bearer {_get_token()}",
        "Content-Type": "application/json",
    }
    url = f"{endpoint}/functions/{function_name}/invoke"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise _UDFCallFailed(
            f"{function_name} {url} -> {e.code} {e.reason}\n{detail}"
        ) from e
    payload = resp.read().decode("utf-8")
    if not payload:
        return {}
    envelope = json.loads(payload)
    status = envelope.get("status")
    if status and status != "Succeeded":
        errors = envelope.get("errors") or envelope
        raise _UDFCallFailed(f"{function_name} {url} -> status={status} errors={errors}")
    return envelope.get("output", envelope)


def _hit_from_wire(d: dict) -> VerifiedHit:
    """Construct a VerifiedHit from a UDF response dict so the existing
    `_format_*` helpers consume it unchanged."""
    return VerifiedHit(
        passage_idx=d["passage_idx"],
        source_file=d["source_file"],
        char_offset=d["char_offset"],
        char_length=d["char_length"],
        content_hash=d["content_hash"],
        drift_status=d["drift_status"],
        score=float(d["score"]),
        text=d["text"],
    )


def cmd_search_fabric(args: argparse.Namespace) -> int:
    """Entry point dispatched from cli/search.py for `fabric://` URLs."""
    from ._fabric_auth import FabricAuthError

    alias, km = _parse_fabric_url(str(args.knowledge_model))
    endpoint = _resolve_alias(alias)
    if endpoint is None:
        print(
            f"error: fabric alias {alias!r} not registered. "
            f"Run `rlat fabric add {alias}=<udf-url>` to register it. "
            f"(Config file: {_config_path()})",
            file=sys.stderr,
        )
        return 2

    try:
        if km is None:
            rows = _post_udf(endpoint, "list_kms", {})
            print(json.dumps(rows, indent=2))
            return 0

        if args.query is None:
            print(
                f"error: query is required for `fabric://{alias}/{km}` search "
                "(only `fabric://<alias>` discovery may omit it)",
                file=sys.stderr,
            )
            return 2

        response = _post_udf(endpoint, "search", {
            "kmName":       km,
            "query":        args.query,
            "topK":         args.top_k,
            "verifiedOnly": args.verified_only,
        })
    except (_UDFCallFailed, FabricAuthError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    hits = [_hit_from_wire(d) for d in response.get("hits", [])]
    band = response.get("band", "base")

    missing_names: list[str] = []
    if args.format == "text":
        print(_format_text(hits))
    elif args.format == "json":
        print(_format_json(hits))
    elif args.format == "context":
        from ..config import MaterialiserConfig
        rendered, missing_names = _format_context(
            hits, MaterialiserConfig(), _grounding.Mode(args.mode),
            band, args.query,
        )
        print(rendered)

    if args.strict_names and missing_names:
        print(
            f"error: --strict-names and distinctive question tokens not "
            f"found in retrieved passages: {','.join(missing_names)}. The "
            f"question may be about an entity the corpus does not cover.",
            file=sys.stderr,
        )
        return 3

    if not args.quiet:
        print(
            f"[search] fabric://{alias}/{km} band={band} hits={len(hits)}",
            file=sys.stderr,
        )
    return 0
