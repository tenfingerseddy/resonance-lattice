"""env_probe — zero-friction, ground-truth capture of user-environment ATTRIBUTES.

The highest-quality / lowest-friction source of band content (capture-frontier charter): a user-environment
attribute that no corpus holds and that lifts answers is, for the most common premise types, MACHINE-READABLE —
so READ it, don't ask. A probe is a deterministic function that turns a real environment signal (a version
command, a config value, the `.rlat`'s own size) into a typed attribute string and lands it in the insight band
via the shipped single-shot `capture_attributes`. Precision is ~1.0 by construction: a read value cannot be
confabulated.

Design: each probe is `(key, signal->text)`. In PRODUCTION the signal comes from a real read (subprocess / file /
authed API); those are platform-gated and live in `_read_signals`. For tests/benchmarks an explicit `signals`
dict is injected (the honest stand-in for "what the real read returned"), so the capture path is exercised with
zero external dependencies. The emitted TEXT is the probe's own natural wording — deliberately not copied from any
fixture — so downstream retrieval is tested against real probe output, not a gold echo.
"""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path
from typing import Callable

from .attribute_capture import capture_attributes

# Each entry: key -> (criticality, fn(signals) -> attribute text | None). A probe returns None when its signal is
# absent (the fact isn't readable on this machine), so the band only ever gets facts that were actually observed.
PowerShellSignals = dict


def _ps_version(s: PowerShellSignals) -> str | None:
    v = s.get("ps_version")
    if not v:
        return None
    ed = s.get("ps_edition")
    name = "Windows PowerShell" if str(v).startswith("5.") else "PowerShell"
    tail = f" ({ed} edition)" if ed else ""
    return f"The user is running {name} {v}{tail}."


def _os(s: PowerShellSignals) -> str | None:
    o = s.get("os")
    return f"The user's operating system is {o}." if o else None


def _account(s: PowerShellSignals) -> str | None:
    a = s.get("account_type")
    if not a:
        return None
    return ("The user's account is a standard (non-administrator) account."
            if a == "standard" else f"The user's account is {a}.")


def _exec_policy(s: PowerShellSignals) -> str | None:
    p = s.get("execution_policy")
    return f"The user's PowerShell execution policy is set to {p}." if p else None


def _domain(s: PowerShellSignals) -> str | None:
    d = s.get("domain_joined")
    if d is None:
        return None
    return ("The user's machine is joined to a corporate Active Directory domain." if d
            else "The user's machine is a standalone workgroup machine (not domain-joined).")


def _language_mode(s: PowerShellSignals) -> str | None:
    m = s.get("language_mode")
    return f"PowerShell is running in {m} mode on the user's machine." if m else None


def _proxy(s: PowerShellSignals) -> str | None:
    p = s.get("proxy")
    if not p:
        return None
    return ("The user's machine reaches the internet only through an authenticated corporate web proxy."
            if p == "authenticated" else f"The user's machine uses a {p} proxy.")


POWERSHELL_PROBES: list[tuple[str, str, Callable[[PowerShellSignals], "str | None"]]] = [
    ("ps_version", "high", _ps_version),
    ("os", "normal", _os),
    ("account_type", "high", _account),
    ("execution_policy", "high", _exec_policy),
    ("domain_joined", "normal", _domain),
    ("language_mode", "high", _language_mode),
    ("proxy", "normal", _proxy),
]


# ---- Microsoft Fabric probes (read from the Fabric/Azure REST APIs the user is already authed to) ----
# A probed attribute disambiguates better for the selector when it carries its salient CONTRAST (a true
# restatement of the read value, not a confabulation) — measured: it recovers role/license selection misses.
_SKU_RANK = ["F2", "F4", "F8", "F16", "F32", "F64", "F128", "F256", "F512", "F1024", "F2048"]
_LOW_ROLES = {"viewer": "a read-only role, not Member or Admin", "contributor": "not Admin"}
_LICENSE_CONTRAST = {"power bi pro": "not Premium Per User (PPU)", "fabric free": "not a paid Pro or PPU license"}


def _fab_sku(s):
    v = s.get("capacity_sku")
    if not v:
        return None
    tail = " (the smallest Fabric capacity)" if v == "F2" else ""
    return f"The user's Fabric capacity is an {v} SKU{tail}."


def _fab_role(s):
    r = s.get("workspace_role")
    if not r:
        return None
    c = _LOW_ROLES.get(r.lower())
    return f"The user's role in the Fabric workspace is {r}" + (f" ({c})." if c else ".")


def _fab_license(s):
    lic = s.get("license")
    if not lic:
        return None
    c = _LICENSE_CONTRAST.get(lic.lower())
    return f"The user has a {lic} license" + (f" ({c})." if c else ".")


def _fab_region(s):
    rg = s.get("capacity_region")
    return f"The user's Fabric capacity is in the {rg} region." if rg else None


def _fab_git(s):
    g = s.get("git_integration")
    if g is None:
        return None
    return ("The tenant admin has disabled Git integration for the user's tenant." if g == "disabled"
            else "Git integration is enabled in the user's tenant.")


def _fab_private_link(s):
    p = s.get("private_link")
    if p is None:
        return None
    return ("The user's tenant has Private Link enabled, blocking public internet access to Fabric." if p
            else "The user's tenant allows public internet access to Fabric.")


def _fab_cap_admin(s):
    a = s.get("capacity_admin")
    if a is None:
        return None
    return ("The user is a Fabric capacity administrator." if a
            else "The user is not a Fabric capacity administrator.")


FABRIC_PROBES = [
    ("capacity_sku", "high", _fab_sku), ("workspace_role", "high", _fab_role),
    ("license", "high", _fab_license), ("capacity_region", "normal", _fab_region),
    ("git_integration", "normal", _fab_git), ("private_link", "normal", _fab_private_link),
    ("capacity_admin", "normal", _fab_cap_admin),
]

PROBE_SETS = {"powershell": POWERSHELL_PROBES, "fabric": FABRIC_PROBES}


def probe_attributes_for(probe_set: str, signals: PowerShellSignals, *, km_path: str | Path | None = None) -> list[str]:
    """Run a named probe set against injected signals. Corpus-general entry for benchmarks / future hosts."""
    probes = PROBE_SETS[probe_set]
    out = [text for _k, _c, fn in probes if (text := fn(signals))]
    if km_path is not None and (cs := corpus_size_attribute(km_path)):
        out.append(cs)
    return out


def corpus_size_attribute(km_path: str | Path) -> str | None:
    """Probe the `.rlat` itself — a zero-external-call attribute (read passages.jsonl line count)."""
    p = Path(km_path)
    try:
        with zipfile.ZipFile(p) as z:
            if "passages.jsonl" not in z.namelist():
                return None
            n = sum(1 for line in z.read("passages.jsonl").splitlines() if line.strip())
        return f"The user's knowledge model contains {n:,} passages."
    except (OSError, zipfile.BadZipFile):
        return None


def _read_signals() -> PowerShellSignals:  # pragma: no cover - platform-gated real reads
    """Best-effort REAL environment read (PowerShell). Each read is independently guarded; a failed read just
    omits that signal. Never raises — a probe that can't read returns nothing."""
    s: PowerShellSignals = {}

    def _ps(expr: str) -> str | None:
        try:
            out = subprocess.run(["pwsh", "-NoProfile", "-Command", expr], capture_output=True,
                                 text=True, timeout=8)
            v = (out.stdout or "").strip()
            return v or None
        except (OSError, subprocess.SubprocessError):
            return None

    s["ps_version"] = _ps("$PSVersionTable.PSVersion.ToString()")
    s["ps_edition"] = _ps("$PSVersionTable.PSEdition")
    s["execution_policy"] = _ps("Get-ExecutionPolicy")
    s["language_mode"] = _ps("$ExecutionContext.SessionState.LanguageMode")
    # OS caption (e.g. 'Windows 11 Pro' / 'Windows 10 Enterprise'); CIM on Windows, $PSVersionTable.OS elsewhere.
    s["os"] = _ps("try { (Get-CimInstance Win32_OperatingSystem).Caption } catch { $PSVersionTable.OS }")
    # Admin vs standard token from the current identity's role membership.
    s["account_type"] = _ps(
        "if (([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent())"
        ".IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {'administrator'} else {'standard'}")
    dj = _ps("try { [string](Get-CimInstance Win32_ComputerSystem).PartOfDomain } catch { '' }")
    if dj in ("True", "False"):
        s["domain_joined"] = (dj == "True")
    # Proxy: WinHTTP default proxy presence (authenticated-vs-not is not reliably readable, so report presence).
    px = _ps("try { ([System.Net.WebRequest]::GetSystemWebProxy()).GetProxy('http://example.com').Host } catch { '' }")
    if px and "example.com" not in px:
        s["proxy"] = "configured"
    return {k: v for k, v in s.items() if v not in (None, "")}


def probe_attributes(signals: PowerShellSignals | None = None, *, km_path: str | Path | None = None) -> list[str]:
    """Run the deterministic probes and return the attribute strings (no capture). Pure given `signals`."""
    s = signals if signals is not None else _read_signals()
    out = [text for _key, _crit, fn in POWERSHELL_PROBES if (text := fn(s))]
    if km_path is not None and (cs := corpus_size_attribute(km_path)):
        out.append(cs)
    return out


def probe_and_capture(km_path: str | Path, signals: PowerShellSignals | None = None,
                      *, include_corpus_size: bool = True, encoder=None) -> list:
    """Probe the environment and land the observed attributes in the `.rlat` insight band (single writeback).

    Returns the minted attribute claims. Zero user friction: nothing is asked; only observed facts are written.
    """
    s = signals if signals is not None else _read_signals()
    # Carry each probe's key (the normalized subject — "ps_version", "capacity_sku", …)
    # as the attribute_key, so the serve-time newest-wins dedup can group by it.
    pairs = [(key, text) for key, _crit, fn in POWERSHELL_PROBES if (text := fn(s))]
    if include_corpus_size and (cs := corpus_size_attribute(km_path)):
        pairs.append(("corpus_size", cs))
    if not pairs:
        return []
    return capture_attributes(km_path, [t for _, t in pairs], keys=[k for k, _ in pairs],
                              criticality="high", encoder=encoder)


def probe_and_capture_for(km_path: str | Path, probe_set: str, signals: PowerShellSignals,
                          *, include_corpus_size: bool = True, encoder=None) -> list:
    """Corpus-general: run a NAMED probe set against injected signals and land the attributes. Same single-shot
    writeback as `probe_and_capture`; used by benchmarks and non-PowerShell hosts."""
    probes = PROBE_SETS[probe_set]
    pairs = [(key, text) for key, _crit, fn in probes if (text := fn(signals))]
    if include_corpus_size and (cs := corpus_size_attribute(km_path)):
        pairs.append(("corpus_size", cs))
    if not pairs:
        return []
    return capture_attributes(km_path, [t for _, t in pairs], keys=[k for k, _ in pairs],
                              criticality="high", encoder=encoder)
