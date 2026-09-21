import json
from pathlib import Path

import pytest

from app.llm import freebuff_byok as fb
from app.llm.freebuff_byok import (
    FreebuffByokError,
    resolve_freebuff_provider,
)


@pytest.fixture()
def byok_env(monkeypatch, tmp_path):
    """Isolate resolver state: no env key, BYOK dir pointed at tmp_path."""
    monkeypatch.delenv("FREEBUFF_API_KEY", raising=False)
    monkeypatch.delenv("FREEBUFF_BYOK_CONNECTION_ID", raising=False)
    monkeypatch.delenv("FREEBUFF_MODEL", raising=False)
    monkeypatch.setenv("FREEBUFF_BYOK_CONFIG_DIR", str(tmp_path))
    yield tmp_path


def _write_connection(tmp_path: Path, **overrides):
    connection = {
        "id": "8a2e1b14-9c12-4c6b-80b9-2280b26729a9",
        "revision": 1,
        "name": "OpenRouter · openrouter",
        "provider": "openrouter",
        "model": "openrouter",
        "baseUrl": "https://openrouter.ai/api/v1",
        "credentialRef": "connection:8a2e1b14-9c12-4c6b-80b9-2280b26729a9:1",
        "createdAt": "2026-09-18T07:11:12.335Z",
        "updatedAt": "2026-09-18T07:11:12.335Z",
    }
    connection.update(overrides)
    (tmp_path / "connections.json").write_text(json.dumps([connection]), encoding="utf-8")
    return connection


def test_env_key_wins_over_byok(monkeypatch, byok_env):
    _write_connection(byok_env)
    monkeypatch.setenv("FREEBUFF_API_KEY", "sk-env-test")
    resolved = resolve_freebuff_provider()
    assert resolved is not None and resolved.source == "env"
    assert resolved.api_key == "sk-env-test"
    assert resolved.base_url == "https://openrouter.ai/api/v1"


def test_env_dead_url_demoted_but_kept_as_last_resort(monkeypatch, byok_env):
    """A nonexistent host (e.g. api.freebuff.ai) must not break the client."""
    monkeypatch.delenv("FREEBUFF_BYOK_CONFIG_DIR", raising=False)  # no BYOK store
    monkeypatch.setenv("FREEBUFF_API_KEY", "sk-env-test")
    monkeypatch.setenv("FREEBUFF_BASE_URL", "https://api.freebuff.ai/v1")
    monkeypatch.setattr(fb, "_host_resolves", lambda url: False)
    monkeypatch.setattr(fb, "_connections_path",
                        lambda: Path("Z:/definitely-missing/connections.json"))
    resolved = resolve_freebuff_provider()
    assert resolved.source == "env"
    # Credential kept, dead endpoint replaced by the Freebuff-supported default.
    assert resolved.api_key == "sk-env-test"
    assert resolved.base_url == fb.FALLBACK_BASE_URL
    assert "does not resolve" in resolved.detail.get("warning", "")


def test_env_dead_url_falls_through_to_byok_without_key(monkeypatch, byok_env):
    """No env key + stale base URL env var => healthy BYOK still wins."""
    _write_connection(byok_env, model="anthropic/claude-sonnet-4")
    monkeypatch.setattr(fb, "_read_secret_via_bun", lambda ref: "sk-or-fake")
    monkeypatch.setenv("FREEBUFF_BASE_URL", "https://api.freebuff.ai/v1")
    monkeypatch.setattr(fb, "_host_resolves", lambda url: False)
    resolved = resolve_freebuff_provider()
    assert resolved.source == "byok"
    assert resolved.model == "anthropic/claude-sonnet-4"


def test_settings_key_used_when_no_env(monkeypatch, byok_env):
    class FakeSettings:
        freebuff_api_key = "sk-file-key"
        freebuff_base_url = "https://openrouter.ai/api/v1"
        freebuff_model = "openai/gpt-4o-mini"

    _write_connection(byok_env)  # would win if settings were ignored
    monkeypatch.setattr(fb, "_read_secret_via_bun", lambda ref: "sk-or-should-not-be-used")
    resolved = resolve_freebuff_provider(settings=FakeSettings())
    assert resolved.source == "settings"
    assert resolved.api_key == "sk-file-key"


def test_settings_dead_url_falls_back_to_byok(monkeypatch, byok_env):
    class DeadSettings:
        freebuff_api_key = "sk-file-key"
        freebuff_base_url = "https://api.freebuff.ai/v1"
        freebuff_model = "freebuff-1"

    _write_connection(byok_env, model="openai/gpt-4.1-mini")
    monkeypatch.setattr(fb, "_read_secret_via_bun", lambda ref: "sk-or-fake")
    monkeypatch.setattr(fb, "_host_resolves", lambda url: False)
    resolved = resolve_freebuff_provider(settings=DeadSettings())
    # Settings credential is unusable for a dead endpoint, so BYOK is used.
    assert resolved.source == "byok"
    assert resolved.model == "openai/gpt-4.1-mini"


def test_loopback_url_skips_dns_check(monkeypatch, byok_env):
    monkeypatch.setenv("FREEBUFF_API_KEY", "sk-env-test")
    monkeypatch.setenv("FREEBUFF_BASE_URL", "http://127.0.0.1:9/v1")
    resolved = resolve_freebuff_provider()
    assert resolved.base_url == "http://127.0.0.1:9/v1"


def test_byok_openrouter_pinned_url_and_model_fallback(monkeypatch, byok_env):
    _write_connection(byok_env)  # model == provider placeholder "openrouter"
    monkeypatch.setattr(fb, "_read_secret_via_bun", lambda ref: "sk-or-fake")
    resolved = resolve_freebuff_provider()
    assert resolved.source == "byok"
    assert resolved.base_url == "https://openrouter.ai/api/v1"
    assert resolved.model == fb.FALLBACK_MODEL
    assert resolved.detail["provider"] == "openrouter"


def test_byok_explicit_model_used(monkeypatch, byok_env):
    _write_connection(byok_env, model="anthropic/claude-sonnet-4")
    monkeypatch.setattr(fb, "_read_secret_via_bun", lambda ref: "sk-or-fake")
    resolved = resolve_freebuff_provider()
    assert resolved.model == "anthropic/claude-sonnet-4"


def test_byok_connection_id_selection(monkeypatch, byok_env):
    _write_connection(byok_env)
    second = _write_connection(
        byok_env,
        id="ffffffff-ffff-ffff-ffff-ffffffffffff",
        name="Second",
        model="openai/gpt-4.1-mini",
    )
    monkeypatch.setenv("FREEBUFF_BYOK_CONNECTION_ID", second["id"])
    monkeypatch.setattr(fb, "_read_secret_via_bun", lambda ref: "sk-or-fake")
    resolved = resolve_freebuff_provider()
    assert resolved.model == "openai/gpt-4.1-mini"


def test_openai_compatible_requires_https(monkeypatch, byok_env):
    _write_connection(
        byok_env,
        provider="openai-compatible",
        baseUrl="http://insecure.example.com/v1",
    )
    monkeypatch.setattr(fb, "_read_secret_via_bun", lambda ref: "sk-x")
    # Resolver surfaces failures as None so the client can report guidance.
    assert resolve_freebuff_provider() is None


def test_openai_compatible_allows_loopback_http(monkeypatch, byok_env):
    _write_connection(
        byok_env,
        provider="openai-compatible",
        model="local-model",
        baseUrl="http://127.0.0.1:8000/v1",
    )
    monkeypatch.setattr(fb, "_read_secret_via_bun", lambda ref: "sk-local")
    resolved = resolve_freebuff_provider()
    assert resolved.base_url == "http://127.0.0.1:8000/v1"


def test_missing_keychain_secret_returns_none(monkeypatch, byok_env):
    _write_connection(byok_env)
    monkeypatch.setattr(fb, "_read_secret_via_bun", lambda ref: "")
    assert resolve_freebuff_provider() is None


def test_missing_connections_file_returns_none(byok_env):
    assert resolve_freebuff_provider() is None


def test_unsupported_provider_returns_none(monkeypatch, byok_env):
    _write_connection(byok_env, provider="mystery-provider")
    assert resolve_freebuff_provider() is None


def test_env_reference_credential(monkeypatch, byok_env):
    _write_connection(
        byok_env,
        provider="openai-compatible",
        model="m",
        baseUrl="https://api.example.com/v1",
        credentialRef="env:MY_CUSTOM_KEY",
    )
    monkeypatch.setenv("MY_CUSTOM_KEY", "sk-from-env-ref")
    resolved = resolve_freebuff_provider()
    assert resolved.api_key == "sk-from-env-ref"


def test_no_secrets_in_detail(monkeypatch, byok_env):
    _write_connection(byok_env)
    monkeypatch.setattr(fb, "_read_secret_via_bun", lambda ref: "sk-or-verysecret")
    resolved = resolve_freebuff_provider()
    assert "verysecret" not in json.dumps(resolved.detail)
