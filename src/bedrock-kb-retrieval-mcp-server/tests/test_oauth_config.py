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

"""Tests for OAuth environment configuration."""

import importlib
import pytest
from unittest.mock import patch


@patch('awslabs.bedrock_kb_retrieval_mcp_server.server.get_bedrock_agent_runtime_client')
@patch('awslabs.bedrock_kb_retrieval_mcp_server.server.get_bedrock_agent_client')
def test_oauth_requires_streamable_http_transport(mock_agent_client, mock_runtime_client, monkeypatch):
    """OAuth enforcement requires Streamable HTTP transport."""
    monkeypatch.setenv('MCP_AUTH_MODE', 'oauth')
    monkeypatch.setenv('MCP_TRANSPORT', 'stdio')
    monkeypatch.setenv('AUTH0_DOMAIN', 'mytenant.us.auth0.com')
    monkeypatch.setenv('AUTH0_AUDIENCE', 'https://example.com/mcp')
    monkeypatch.setenv('MCP_RESOURCE_URL', 'https://example.com/mcp')

    import awslabs.bedrock_kb_retrieval_mcp_server.server

    with pytest.raises(ValueError, match='MCP_TRANSPORT=streamable-http'):
        importlib.reload(awslabs.bedrock_kb_retrieval_mcp_server.server)


@patch('awslabs.bedrock_kb_retrieval_mcp_server.server.get_bedrock_agent_runtime_client')
@patch('awslabs.bedrock_kb_retrieval_mcp_server.server.get_bedrock_agent_client')
def test_oauth_requires_domain_audience_and_resource_url(mock_agent_client, mock_runtime_client, monkeypatch):
    """OAuth requires Auth0 env vars and resource URL."""
    monkeypatch.setenv('MCP_AUTH_MODE', 'oauth')
    monkeypatch.setenv('MCP_TRANSPORT', 'streamable-http')

    import awslabs.bedrock_kb_retrieval_mcp_server.server

    with pytest.raises(ValueError, match='AUTH0_DOMAIN is required'):
        importlib.reload(awslabs.bedrock_kb_retrieval_mcp_server.server)


@patch('awslabs.bedrock_kb_retrieval_mcp_server.server.get_bedrock_agent_runtime_client')
@patch('awslabs.bedrock_kb_retrieval_mcp_server.server.get_bedrock_agent_client')
def test_oauth_configures_fastmcp_auth(mock_agent_client, mock_runtime_client, monkeypatch):
    """OAuth mode should configure FastMCP auth and token verification."""
    monkeypatch.setenv('MCP_AUTH_MODE', 'oauth')
    monkeypatch.setenv('MCP_TRANSPORT', 'streamable-http')
    monkeypatch.setenv('AUTH0_DOMAIN', 'mytenant.us.auth0.com')
    monkeypatch.setenv('AUTH0_AUDIENCE', 'https://example.com/mcp')
    monkeypatch.setenv('MCP_RESOURCE_URL', 'https://example.com/mcp')

    import awslabs.bedrock_kb_retrieval_mcp_server.server

    mod = importlib.reload(awslabs.bedrock_kb_retrieval_mcp_server.server)
    assert mod.mcp.settings.auth is not None
