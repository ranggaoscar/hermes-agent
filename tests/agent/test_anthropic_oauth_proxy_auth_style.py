"""A pooled OAuth credential behind a transparent first-party proxy keeps its OAuth wire identity (#148400).

``_auth_style`` classified any non-anthropic.com base_url as a third-party endpoint *before* the
OAuth shape test and never saw the provider, so an ``sk-ant-oat…`` credential routed through a
local Anthropic-Messages-protocol proxy went out as ``x-api-key`` — the first-party API 401s it,
even though ``anthropic_route_is_oauth`` (the same decision, another function, #114967) already
answers OAuth for the ``anthropic`` provider regardless of base_url. Genuine third-party keys
keep ``x-api-key``: only a credential that is *already* an OAuth token on a first-party
``anthropic`` route flips to Bearer + Claude Code identity.
"""

from __future__ import annotations

import pytest

from agent.anthropic_adapter import _auth_style

# Synthetic credentials shaped like the real families (never live secrets).
OAUTH_TOKEN = "sk-" + "ant-oat01-" + "X" * 24  # Claude Code OAuth access token
CONSOLE_KEY = "sk-" + "ant-api03-" + "X" * 24  # Console API key: pays per token, x-api-key
THIRD_PARTY_KEY = "third-party-proxy-key"

PROXY = "http://10.100.168.1:8787"  # transparent local Anthropic-Messages-protocol proxy


def wire_headers(client) -> dict:
    """Headers the SDK would put on a /v1/messages POST (the Omit() default is resolved here)."""
    from anthropic._models import FinalRequestOptions

    return dict(client._build_headers(FinalRequestOptions(method="post", url="/v1/messages", json_data={})))


@pytest.mark.parametrize(("key", "base_url", "expected"), [
    # No-provider calls: the legacy classification must be identical before and after the fix.
    (OAUTH_TOKEN, "https://api.anthropic.com", "oauth"),
    (OAUTH_TOKEN, "", "oauth"),  # empty base_url is the native default route
    (CONSOLE_KEY, "https://api.anthropic.com", "api_key"),
    (OAUTH_TOKEN, PROXY, "api_key"),  # no provider context: unchanged legacy behaviour
    (THIRD_PARTY_KEY, PROXY, "api_key"),
    (OAUTH_TOKEN, "https://api.minimax.io/anthropic", "bearer"),
    (OAUTH_TOKEN, "https://api.kimi.com/coding", "kimi"),
])
def test_auth_style_legacy_calls_unchanged(key, base_url, expected):
    assert _auth_style(key, base_url, base_url) == expected


@pytest.mark.parametrize(("key", "base_url", "provider", "expected"), [
    # #148400: the anthropic provider behind a transparent proxy keeps the OAuth wire identity.
    (OAUTH_TOKEN, PROXY, "anthropic", "oauth"),
    (OAUTH_TOKEN, "", "anthropic", "oauth"),
    # A Console key behind the same proxy is still an x-api-key credential (third-party
    # contract kept — the proxy gets the operator's own key, not subscription identity).
    (CONSOLE_KEY, PROXY, "anthropic", "api_key"),
    (CONSOLE_KEY, "https://api.anthropic.com", "anthropic", "api_key"),
    # Named third-party providers keep their own key/auth style whatever the credential shape.
    (THIRD_PARTY_KEY, PROXY, "enterprise-relay", "api_key"),
    (OAUTH_TOKEN, "https://api.minimax.io/anthropic", "minimax-oauth", "bearer"),
    (OAUTH_TOKEN, "https://api.kimi.com/coding", "kimi", "kimi"),
])
def test_auth_style_provider_classification(key, base_url, provider, expected):
    assert _auth_style(key, base_url, base_url, provider=provider) == expected


def test_oauth_credential_through_proxy_keeps_claude_code_identity(monkeypatch):
    """End-to-end through build_anthropic_client: Bearer + Claude Code identity, no x-api-key."""
    pytest.importorskip("anthropic")
    from agent.anthropic_adapter import _OAUTH_ONLY_BETAS, build_anthropic_client

    monkeypatch.setattr("agent.anthropic_adapter._claude_code_version_cache", "0.0.0")
    client = build_anthropic_client(OAUTH_TOKEN, PROXY, provider="anthropic")
    headers = {k.lower(): v for k, v in wire_headers(client).items()}
    assert headers.get("authorization") == "Bearer " + OAUTH_TOKEN
    assert "x-api-key" not in headers
    assert "claude-code/" in headers.get("user-agent", "")
    assert headers.get("x-app") == "cli"
    assert all(beta in headers.get("anthropic-beta", "") for beta in _OAUTH_ONLY_BETAS)


def test_third_party_key_through_proxy_still_uses_x_api_key():
    """Control: a plain third-party key behind the same proxy keeps x-api-key, no identity headers."""
    pytest.importorskip("anthropic")
    from agent.anthropic_adapter import build_anthropic_client

    client = build_anthropic_client(THIRD_PARTY_KEY, PROXY)
    headers = {k.lower(): v for k, v in wire_headers(client).items()}
    assert headers.get("x-api-key") == THIRD_PARTY_KEY
    assert "authorization" not in headers
    assert "claude-code/" not in headers.get("user-agent", "")
