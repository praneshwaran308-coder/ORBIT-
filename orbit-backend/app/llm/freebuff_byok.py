"""Freebuff BYOK provider resolution.

Freebuff Desktop stores "BYOK" (bring-your-own-key) connections in
``~/.config/freebuff/byok/connections.json`` and keeps each connection's
credential in the OS keychain via ``Bun.secrets`` under the service
``com.freebuff.byok.v1`` (name = the connection's ``credentialRef``).

This module resolves a usable provider configuration from that store —
the same mechanism Freebuff's own orchestrator uses — so ORBIT can talk to
the configured provider through Freebuff without any keys being hardcoded,
printed, or copied. Explicit environment variables always win when present.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger("orbit.freebuff")

BYOK_SECRET_SERVICE = "com.freebuff.byok.v1"
ENV_REFERENCE = re.compile(r"^env:([A-Za-z_][A-Za-z0-9_]*)$")
OWNED_REFERENCE = re.compile(r"^connection:[a-f0-9-]+:[1-9][0-9]*$")

# Freebuff's supported integration is the BYOK provider endpoint; these
# constants are the conservative defaults used when neither env vars nor a
# usable BYOK connection supply a concrete value.
FALLBACK_BASE_URL = "https://openrouter.ai/api/v1"
# Used only when a connection's `model` field is the provider placeholder
# (e.g. "openrouter") rather than a concrete model id. Override with
# FREEBUFF_MODEL at any time.
FALLBACK_MODEL = "openai/gpt-4o-mini"

_BUN_SUBPROCESS_TIMEOUT_SECONDS = 15


@dataclass
class ResolvedProvider:
    base_url: str
    api_key: str
    model: str
    source: str                       # "env" | "byok"
    detail: dict[str, str]            # secret-free provenance info


class FreebuffByokError(RuntimeError):
    """Raised when the Freebuff BYOK store cannot provide a usable provider."""


# -- public API -------------------------------------------------------------


def resolve_freebuff_provider(settings=None) -> ResolvedProvider | None:
    """Resolve provider settings, in strict order:

      1. Environment variables (FREEBUFF_API_KEY / _BASE_URL / _MODEL)
      2. Settings file values (orbit-backend/.data/.env via pydantic-settings)
      3. Freebuff Desktop's BYOK store (connections.json + OS keychain)

    A source whose base URL is unusable (bad scheme, or a host that does not
    resolve — e.g. the fictional api.freebuff.ai) is demoted to *last resort*:
    its credential is kept but paired with the Freebuff-supported default
    endpoint, and it is only used when no healthy source exists. Returns
    None when nothing is usable.
    """
    last_resort: ResolvedProvider | None = None
    for candidate, fallback in (_env_provider(), _settings_provider(settings)):
        if candidate is not None:
            return candidate
        if fallback is not None and last_resort is None:
            last_resort = fallback

    try:
        byok = _resolve_from_byok_store()
    except FreebuffByokError as exc:
        logger.info("Freebuff BYOK resolution unavailable: %s", exc)
        byok = None
    if byok is not None:
        return byok
    return last_resort


def _credential_provider(api_key: str, base_url: str, model: str,
                         source: str, via: str) -> tuple[ResolvedProvider | None,
                                                         ResolvedProvider | None]:
    """Shared credential-source logic with URL sanity checking.

    Returns (healthy_provider, last_resort_provider); at most one is set.
    """
    base_url = (base_url or "").strip().rstrip("/")
    warning = ""
    if base_url:
        if not _is_acceptable_base_url(base_url):
            warning = (
                f"Ignoring configured base URL '{base_url}': not an acceptable "
                "OpenAI-compatible endpoint (HTTPS required; HTTP only on "
                "loopback)."
            )
            base_url = ""
        elif base_url and not _is_loopback_url(base_url) and not _host_resolves(base_url):
            warning = (
                f"Configured base URL '{base_url}' does not resolve in DNS (the "
                "endpoint does not exist). Freebuff's supported integration is "
                "the BYOK provider endpoint."
            )
            base_url = ""
        if warning:
            logger.warning("%s", warning)

    detail: dict[str, str] = {"via": via}
    if warning:
        detail["warning"] = warning

    provider = ResolvedProvider(
        base_url=base_url or FALLBACK_BASE_URL,
        api_key=api_key,
        model=(model or "").strip() or FALLBACK_MODEL,
        source=source,
        detail=detail,
    )
    return (provider, None) if not warning else (None, provider)


def _env_provider() -> tuple[ResolvedProvider | None, ResolvedProvider | None]:
    """Build a provider from explicit environment variables."""
    env_key = os.environ.get("FREEBUFF_API_KEY", "").strip()
    if not env_key:
        return None, None
    return _credential_provider(
        env_key,
        os.environ.get("FREEBUFF_BASE_URL", ""),
        os.environ.get("FREEBUFF_MODEL", ""),
        source="env",
        via="FREEBUFF_API_KEY environment variable",
    )


def _settings_provider(settings) -> tuple[ResolvedProvider | None, ResolvedProvider | None]:
    """Build a provider from Settings (e.g. orbit-backend/.data/.env)."""
    if settings is None:
        return None, None
    key = (getattr(settings, "freebuff_api_key", "") or "").strip()
    if not key:
        return None, None
    return _credential_provider(
        key,
        getattr(settings, "freebuff_base_url", "") or "",
        getattr(settings, "freebuff_model", "") or "",
        source="settings",
        via="FREEBUFF_API_KEY in settings file (.data/.env)",
    )


def _is_loopback_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in ("localhost", "[::1]") or host.startswith("127.0.0.1")


def _host_resolves(url: str) -> bool:
    host = urlparse(url).hostname or ""
    if not host:
        return False
    try:
        socket.getaddrinfo(host, None)
        return True
    except socket.gaierror:
        return False


# -- BYOK store ---------------------------------------------------------------


def _connections_path() -> Path:
    override = os.environ.get("FREEBUFF_BYOK_CONFIG_DIR", "").strip()
    base = Path(override) if override else Path.home() / ".config" / "freebuff" / "byok"
    return base / "connections.json"


def _load_connections() -> list[dict]:
    path = _connections_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FreebuffByokError(f"BYOK connections file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FreebuffByokError(f"BYOK connections file is corrupt: {exc}") from exc

    if isinstance(raw, dict) and isinstance(raw.get("connections"), list):
        raw = raw["connections"]
    if not isinstance(raw, list):
        raise FreebuffByokError("BYOK connections file has an unexpected shape.")
    return [c for c in raw if isinstance(c, dict)]


def _select_connection(connections: list[dict]) -> dict:
    wanted = os.environ.get("FREEBUFF_BYOK_CONNECTION_ID", "").strip()
    if wanted:
        for c in connections:
            if c.get("id") == wanted:
                return c
        raise FreebuffByokError(f"No BYOK connection with id '{wanted}'.")
    if not connections:
        raise FreebuffByokError("No Freebuff BYOK connections are configured.")
    return connections[0]


def _base_url_for(connection: dict) -> str:
    provider = connection.get("provider", "")
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"  # Freebuff pins OpenRouter's URL
    if provider == "openai-compatible":
        url = (connection.get("baseUrl") or "").strip().rstrip("/")
        if not _is_acceptable_base_url(url):
            raise FreebuffByokError(
                "BYOK connection needs a valid HTTPS baseUrl "
                "(HTTP is allowed only on loopback)."
            )
        return url
    raise FreebuffByokError(f"Unsupported BYOK provider '{provider or '(missing)'}'.")


def _is_acceptable_base_url(url: str) -> bool:
    # Mirrors Freebuff's own rule: HTTPS always; HTTP only on loopback.
    match = re.match(r"^(https?)://([^/?#]+)", url)
    if not match:
        return False
    scheme, host = match.group(1), match.group(2).lower()
    loopback = host in ("localhost", "[::1]") or host.startswith("127.0.0.1")
    return scheme == "https" or loopback


def _model_for(connection: dict) -> str:
    model = (connection.get("model") or "").strip()
    if not model or model == connection.get("provider"):
        return os.environ.get("FREEBUFF_MODEL", "").strip() or FALLBACK_MODEL
    return model


def _resolve_credential(connection: dict) -> str:
    ref = (connection.get("credentialRef") or "").strip()
    env_match = ENV_REFERENCE.match(ref)
    if env_match:
        value = os.environ.get(env_match.group(1), "").strip()
        if not value:
            raise FreebuffByokError(
                f"Credential env var '{env_match.group(1)}' is not set."
            )
        return value
    if not OWNED_REFERENCE.match(ref):
        raise FreebuffByokError(f"Unsupported credentialRef format: '{ref[:24]}…'.")
    secret = _read_secret_via_bun(ref)
    if not secret:
        raise FreebuffByokError(
            "OS keychain has no credential for this BYOK connection "
            f"({ref[:24]}…). Open Freebuff Desktop and re-enter the key."
        )
    return secret


def _read_secret_via_bun(reference: str) -> str:
    bun = _find_bun()
    if bun is None:
        raise FreebuffByokError(
            "Could not locate Freebuff's bundled Bun runtime (needed to read "
            "the OS keychain). Set FREEBUFF_BUN_PATH to bun.exe."
        )
    script = (
        "const s = await Bun.secrets.get({"
        "service: process.env.ORBIT_BYOK_SERVICE, "
        "name: process.env.ORBIT_BYOK_REF});"
        "if (s) console.log(s);"
    )
    try:
        proc = subprocess.run(
            [str(bun), "-e", script],
            capture_output=True,
            text=True,
            timeout=_BUN_SUBPROCESS_TIMEOUT_SECONDS,
            env={
                **os.environ,
                "ORBIT_BYOK_SERVICE": BYOK_SECRET_SERVICE,
                "ORBIT_BYOK_REF": reference,
            },
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise FreebuffByokError(f"Failed to query the OS keychain: {exc}") from exc
    if proc.returncode != 0:
        raise FreebuffByokError("Could not unlock the BYOK credential store.")
    secret = (proc.stdout or "").strip()
    if "\r" in secret or "\n" in secret:
        raise FreebuffByokError("BYOK credential is unavailable or invalid.")
    return secret


def _find_bun() -> Path | None:
    override = os.environ.get("FREEBUFF_BUN_PATH", "").strip()
    if override:
        p = Path(override)
        return p if p.is_file() else None
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        Path(local) / "Programs" / "@codebufffreebuff-desktop" / "resources" / "bun" / "bun.exe"
        if local else None,
        Path("C:/Program Files/@codebufffreebuff-desktop/resources/bun/bun.exe"),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    which = None
    try:
        which = subprocess.run(
            ["where", "bun"] if os.name == "nt" else ["which", "bun"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
    except (subprocess.TimeoutExpired, OSError):
        pass
    for line in which or []:
        p = Path(line.strip())
        if p.is_file():
            return p
    return None


def _resolve_from_byok_store() -> ResolvedProvider:
    connections = _load_connections()
    connection = _select_connection(connections)
    return ResolvedProvider(
        base_url=_base_url_for(connection),
        api_key=_resolve_credential(connection),
        model=_model_for(connection),
        source="byok",
        detail={
            "via": "Freebuff BYOK connection",
            "connection_name": str(connection.get("name", "")),
            "connection_id": str(connection.get("id", "")),
            "provider": str(connection.get("provider", "")),
        },
    )
