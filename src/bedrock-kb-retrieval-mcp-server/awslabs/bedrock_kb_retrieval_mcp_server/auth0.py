# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Auth0 OAuth helpers for FastMCP (Streamable HTTP).

This module implements token verification against Auth0 JWKS for use with
FastMCP's `token_verifier` hook.
"""

from __future__ import annotations

import jwt
from dataclasses import dataclass
from jwt.jwks_client import PyJWKClient
from mcp.server.auth.provider import AccessToken, TokenVerifier


@dataclass(frozen=True)
class Auth0Config:
    """Auth0 configuration for JWT verification."""

    domain: str
    audience: str

    @property
    def issuer(self) -> str:
        """Return the expected issuer string for Auth0 tokens."""
        return f'https://{self.domain}/'

    @property
    def jwks_url(self) -> str:
        """Return the Auth0 JWKS URL for this tenant."""
        return f'https://{self.domain}/.well-known/jwks.json'


class Auth0JWTTokenVerifier(TokenVerifier):
    """Verify Auth0-issued RS256 JWTs using JWKS."""

    def __init__(
        self,
        *,
        config: Auth0Config,
        jwks_cache_seconds: int = 3600,
        http_timeout_seconds: int = 5,
    ):
        """Create a token verifier.

        Args:
            config: Auth0 tenant config.
            jwks_cache_seconds: How long to cache JWKS before refreshing.
            http_timeout_seconds: Timeout for fetching JWKS.
        """
        self._config = config
        jwks_cache_seconds = max(60, int(jwks_cache_seconds))
        http_timeout_seconds = max(1, int(http_timeout_seconds))
        self._jwks_client = PyJWKClient(
            self._config.jwks_url,
            cache_jwk_set=True,
            lifespan=jwks_cache_seconds,
            timeout=http_timeout_seconds,
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        """Verify a bearer token and return access info if valid."""
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                key=signing_key.key,
                algorithms=['RS256'],
                audience=self._config.audience,
                issuer=self._config.issuer,
                options={
                    'require': ['exp', 'iat'],
                },
            )

            scopes: list[str] = []
            scope_raw = payload.get('scope')
            if isinstance(scope_raw, str) and scope_raw.strip():
                scopes = [s for s in scope_raw.split(' ') if s]

            client_id = None
            for key in ('azp', 'client_id', 'sub'):
                val = payload.get(key)
                if isinstance(val, str) and val.strip():
                    client_id = val.strip()
                    break
            if not client_id:
                client_id = 'unknown'

            exp = payload.get('exp')
            expires_at = int(exp) if isinstance(exp, (int, float)) else None

            return AccessToken(
                token=token,
                client_id=client_id,
                scopes=scopes,
                expires_at=expires_at,
                resource=self._config.audience,
            )
        except Exception:
            return None
