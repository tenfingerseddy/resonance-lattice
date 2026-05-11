"""`rlat fabric <subcommand>` — manage Fabric UDF aliases + skill scaffold.

  rlat fabric add team=<udf-url>     — register an alias, scaffold the skill
  rlat fabric list                    — print registered aliases
  rlat fabric remove <alias>          — drop an alias

Aliases live at `~/.config/rlat/fabric.toml` (override via `RLAT_FABRIC_CONFIG`).
The skill scaffold lands at `<cwd>/.claude/skills/rlat-fabric-search/SKILL.md`
on first `add` and is rewritten on every subsequent `add`/`remove` so its
frontmatter description always lists the currently registered aliases.

Read by:
  - `rlat search fabric://<alias>[/<km>]` (cli/_fabric.py)
  - The skill scaffold's `!command` block

Skill scaffold is idempotent — safe to run `rlat fabric add` repeatedly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._fabric import _config_path, _load_aliases

EXIT_OK = 0
EXIT_USAGE = 2

SKILL_REL_PATH = Path(".claude/skills/rlat-fabric-search/SKILL.md")


def _atomic_write_toml(path: Path, aliases: dict[str, dict]) -> None:
    """Serialise `aliases` to TOML and atomic-replace `path`.

    Atomic to prevent half-written configs on Ctrl-C — `_load_aliases`
    will read either the old or new file, never a truncated one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for alias in sorted(aliases):
        entry = aliases[alias]
        lines.append(f"[aliases.{alias}]")
        for key in sorted(entry):
            value = entry[key]
            quoted = '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'
            lines.append(f'{key} = {quoted}')
        lines.append("")
    body = "\n".join(lines).rstrip() + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def _scaffold_skill(cwd: Path, aliases: dict[str, dict]) -> Path:
    """Write `<cwd>/.claude/skills/rlat-fabric-search/SKILL.md` (idempotent).

    Skill frontmatter description always reflects the current alias set so
    Claude's skill-trigger heuristics see the live list.
    """
    target = cwd / SKILL_REL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    if not aliases:
        # No aliases registered → write a placeholder skill that points
        # the user at `rlat fabric add`. Still registered so `/skill` lists
        # it; the command body refuses cleanly with a hint.
        alias_list = "(no aliases registered yet)"
        commands_block = (
            "echo \"error: no fabric aliases registered. "
            "Run 'rlat fabric add <name>=<url>' first.\" >&2; exit 2"
        )
    else:
        alias_list = ", ".join(sorted(aliases))
        # Pick the alphabetically-first alias as the example default.
        default_alias = sorted(aliases)[0]
        commands_block = (
            "# Pass --alias=<name> to target a specific Fabric UDF; "
            f"defaults to {default_alias!r}.\n"
            "ALIAS=\"${1:-" + default_alias + "}\"\n"
            "QUERY=\"$2\"\n"
            "if [ -z \"$QUERY\" ]; then\n"
            "    rlat search fabric://$ALIAS\n"
            "else\n"
            "    rlat search fabric://$ALIAS \"$QUERY\" --format=context\n"
            "fi"
        )

    body = (
        "---\n"
        "name: rlat-fabric-search\n"
        f"description: Search Fabric-hosted rlat knowledge models. Registered aliases: {alias_list}. "
        "Trigger when the user asks to search team docs, runbooks, internal references, or any "
        "corpus hosted via a Fabric UDF.\n"
        "---\n\n"
        "# rlat-fabric-search\n\n"
        "Searches Fabric-hosted `.rlat` knowledge models registered as `fabric://` aliases.\n\n"
        f"Registered aliases: **{alias_list}**\n\n"
        "## How to use\n\n"
        "```bash\n"
        "!command\n"
        f"{commands_block}\n"
        "```\n\n"
        "## Discovery\n\n"
        "Run with no query to list available knowledge models on a Fabric UDF:\n\n"
        "```\n"
        "rlat search fabric://<alias>\n"
        "```\n\n"
        "## Adding a new alias\n\n"
        "```\n"
        "rlat fabric add <alias>=<udf-endpoint-url>\n"
        "```\n\n"
        "Re-running `rlat fabric add` rewrites this file so the alias list above stays current.\n"
    )
    target.write_text(body, encoding="utf-8")
    return target


def _cwd(args: argparse.Namespace) -> Path:
    return Path(args.cwd) if args.cwd else Path.cwd()


# ---------- subcommands -----------------------------------------------------

def _cmd_add(args: argparse.Namespace) -> int:
    spec = args.alias_spec
    if "=" not in spec:
        print(f"error: expected '<alias>=<url>', got {spec!r}", file=sys.stderr)
        return EXIT_USAGE
    alias, url = spec.split("=", 1)
    alias, url = alias.strip(), url.strip()
    if not alias or not url:
        print(f"error: empty alias or url in {spec!r}", file=sys.stderr)
        return EXIT_USAGE

    aliases = _load_aliases()
    aliases[alias] = {"url": url}
    _atomic_write_toml(_config_path(), aliases)

    skill_path = _scaffold_skill(_cwd(args), aliases)

    print(f"registered fabric://{alias} -> {url}")
    print(f"  config: {_config_path()}")
    print(f"  skill:  {skill_path}")
    return EXIT_OK


def _cmd_list(args: argparse.Namespace) -> int:  # noqa: ARG001
    aliases = _load_aliases()
    if not aliases:
        print(f"(no aliases registered — config: {_config_path()})")
        return EXIT_OK
    width = max(len(a) for a in aliases)
    for alias in sorted(aliases):
        url = aliases[alias].get("url", "(no url)")
        print(f"  {alias.ljust(width)}  {url}")
    return EXIT_OK


def _cmd_remove(args: argparse.Namespace) -> int:
    aliases = _load_aliases()
    if args.alias not in aliases:
        print(f"error: fabric alias {args.alias!r} not registered", file=sys.stderr)
        return EXIT_USAGE
    del aliases[args.alias]
    _atomic_write_toml(_config_path(), aliases)
    skill_path = _scaffold_skill(_cwd(args), aliases)
    print(f"removed fabric://{args.alias}")
    print(f"  config: {_config_path()}")
    print(f"  skill:  {skill_path}")
    return EXIT_OK


def _cmd_publish_rlat(args: argparse.Namespace) -> int:
    """Upload a local .rlat to a Fabric Lakehouse.

    Uses azure-identity DefaultAzureCredential — picks up SP env vars
    (AZURE_CLIENT_ID / SECRET / TENANT_ID) when set, falls back to
    Azure CLI / interactive sign-in otherwise.

    Storage scope is `https://storage.azure.com/.default` — different from
    the `analysis.windows.net/powerbi/api/.default` scope used to invoke
    UDFs. DefaultAzureCredential handles that automatically.

    Discovery is dynamic — the UDF lists `Files/<km-dir>/` live, so any
    upload (this helper, drag-drop, or `notebookutils.fs.cp`) shows up
    on the next `list_kms` call.
    """
    src = Path(args.rlat_path).resolve()
    if not src.exists():
        print(f"error: {src} not found", file=sys.stderr)
        return EXIT_USAGE
    if not src.suffix == ".rlat":
        print(f"error: expected a .rlat file, got {src.name!r}", file=sys.stderr)
        return EXIT_USAGE

    km_dir = (args.km_dir or "rlat").strip("/")
    km_name = args.km_name or src.stem

    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.filedatalake import DataLakeServiceClient
    except ImportError:
        print(
            "error: `rlat fabric publish-rlat` requires the [fabric] extra. "
            "Install with: pip install 'rlat[fabric]'",
            file=sys.stderr,
        )
        return EXIT_USAGE

    cred = DefaultAzureCredential()
    svc = DataLakeServiceClient(
        account_url="https://onelake.dfs.fabric.microsoft.com",
        credential=cred,
    )
    fs = svc.get_file_system_client(args.workspace)

    # overwrite=True so re-publishes work; the bootstrap mtime cache-bust
    # handles in-flight UDF callers.
    rlat_path = f"{args.lakehouse}/Files/{km_dir}/{km_name}.rlat"
    fc = fs.get_file_client(rlat_path)
    size = src.stat().st_size
    print(f"uploading {src.name} ({size / 1e6:.1f} MB) -> {rlat_path}")
    with src.open("rb") as f:
        fc.upload_data(f, overwrite=True, length=size)
    print(f"published {km_name!r}; next list_kms will surface it")
    return EXIT_OK


# ---------- parser wiring ---------------------------------------------------

def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "fabric",
        help="manage Fabric UDF aliases (rlat search fabric://...)",
        description="Register, list, or remove `fabric://<alias>` aliases for "
                    "Fabric-hosted .rlat knowledge models. Each alias maps to a "
                    "UDF endpoint URL; the `rlat-fabric-search` skill is "
                    "scaffolded into the cwd's .claude/skills/ on add/remove.",
    )
    p.add_argument(
        "--cwd", default=None,
        help="override the cwd used for the skill scaffold (default: $PWD)",
    )
    fab_sub = p.add_subparsers(dest="fabric_command", required=True)

    add_p = fab_sub.add_parser("add", help="register an alias and scaffold the skill")
    add_p.add_argument("alias_spec", help="<alias>=<udf-endpoint-url>")
    add_p.set_defaults(func=_cmd_add)

    list_p = fab_sub.add_parser("list", help="show registered aliases")
    list_p.set_defaults(func=_cmd_list)

    remove_p = fab_sub.add_parser("remove", help="drop an alias")
    remove_p.add_argument("alias", help="alias to drop")
    remove_p.set_defaults(func=_cmd_remove)

    publish_p = fab_sub.add_parser(
        "publish-rlat",
        help="upload a local .rlat to a Fabric Lakehouse",
        description="Maintainer helper: upload a .rlat to "
                    "Files/<km-dir>/<km-name>.rlat in the named Lakehouse. "
                    "UDF discovery lists the directory live, so the new KM "
                    "surfaces on the next list_kms call. Uses "
                    "DefaultAzureCredential (SP env vars / Azure CLI / device code).",
    )
    publish_p.add_argument("rlat_path", help="local .rlat path to upload")
    publish_p.add_argument("--workspace", required=True, help="Fabric workspace ID (GUID)")
    publish_p.add_argument("--lakehouse", required=True, help="Lakehouse item ID (GUID)")
    publish_p.add_argument("--km-dir", default=None,
                           help="Files/<dir>/ holding the .rlats (default: rlat)")
    publish_p.add_argument("--km-name", default=None,
                           help="KM name (default: rlat stem)")
    publish_p.set_defaults(func=_cmd_publish_rlat)
