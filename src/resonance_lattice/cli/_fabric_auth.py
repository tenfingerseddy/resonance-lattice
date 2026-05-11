"""Microsoft Entra token minting for `rlat search fabric://...`.

Service-principal env vars (`AZURE_CLIENT_ID/SECRET/TENANT_ID`) take
priority when present; device-code flow otherwise. Tokens are cached
to the OS keyring via msal-extensions when available, in-memory
otherwise. Plain-language docs in `docs/user/FABRIC.md`.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# UDF *invocation* uses the Power BI scope. The Fabric API scope
# (`https://api.fabric.microsoft.com/.default`) is for item CRUD only —
# invoking with it returns 401.
TOKEN_SCOPE = "https://analysis.windows.net/powerbi/api/.default"
TOKEN_CACHE_NAME = "rlat-fabric"

_SP_ENV_VARS = ("AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID")

# Process-cached credential, keyed by env-signature so a mid-session
# env change rebinds.
_CREDENTIAL: tuple[tuple[str, ...], Any] | None = None


class FabricAuthError(RuntimeError):
    """Token minting failed; caller turns it into stderr + a non-zero rc."""


def _have_sp_env() -> bool:
    return all(os.environ.get(v) for v in _SP_ENV_VARS)


def _env_signature() -> tuple[str, ...]:
    return tuple(os.environ.get(v, "") for v in _SP_ENV_VARS)


def _make_credential() -> Any:
    try:
        from azure.identity import (
            ClientSecretCredential,
            DeviceCodeCredential,
        )
    except ImportError as e:
        raise FabricAuthError(
            "`rlat search fabric://...` requires the [fabric] extra. "
            "Install with: pip install 'rlat[fabric]'"
        ) from e

    if _have_sp_env():
        return ClientSecretCredential(
            tenant_id=os.environ["AZURE_TENANT_ID"],
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"],
        )

    # Persistent keyring cache when msal-extensions is available; in-memory otherwise.
    try:
        from azure.identity import TokenCachePersistenceOptions
        cache_opts = TokenCachePersistenceOptions(
            name=TOKEN_CACHE_NAME, allow_unencrypted_storage=True,
        )
        return DeviceCodeCredential(
            prompt_callback=_prompt_callback,
            cache_persistence_options=cache_opts,
        )
    except (ImportError, Exception):  # noqa: BLE001
        return DeviceCodeCredential(prompt_callback=_prompt_callback)


def _prompt_callback(verification_uri: str, user_code: str, expires_on) -> None:  # noqa: ARG001
    print("[rlat fabric] sign in to Microsoft Entra:", file=sys.stderr)
    print(f"  open {verification_uri} and enter code: {user_code}", file=sys.stderr)


def _credential() -> Any:
    global _CREDENTIAL
    sig = _env_signature()
    if _CREDENTIAL is None or _CREDENTIAL[0] != sig:
        _CREDENTIAL = (sig, _make_credential())
    return _CREDENTIAL[1]


def get_token() -> str:
    try:
        return _credential().get_token(TOKEN_SCOPE).token
    except Exception as e:  # noqa: BLE001
        raise FabricAuthError(f"could not mint Fabric token: {e}") from e
