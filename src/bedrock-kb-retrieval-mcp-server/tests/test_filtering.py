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

import pytest
from awslabs.bedrock_kb_retrieval_mcp_server.knowledgebases.filters import (
    FilterError,
    build_where_filter,
    validate_raw_filter_against_schema,
)
from awslabs.bedrock_kb_retrieval_mcp_server.knowledgebases.schema import (
    AliasSchema,
    MetadataFieldSchema,
    MetadataSchemaFile,
)


def _collect_filter_clauses(node):
    if not isinstance(node, dict):
        return []
    if 'andAll' in node:
        out = []
        for child in node.get('andAll') or []:
            out.extend(_collect_filter_clauses(child))
        return out
    if 'orAll' in node:
        out = []
        for child in node.get('orAll') or []:
            out.extend(_collect_filter_clauses(child))
        return out
    return [node]


class TestWhereFilter:
    """Tests for building schema-driven `where` filters."""

    def test_build_where_filter_and(self):
        """Build an AND-composed where filter with an alias + date range operator-map."""
        schema = MetadataSchemaFile(
            metadata={
                'entity_ids': MetadataFieldSchema(type='STRING'),
                'date': MetadataFieldSchema(type='STRING'),
            },
            aliases={
                'record_id': AliasSchema(
                    metadata_key='entity_ids',
                    operator='stringContains',
                    token_prefix='record_id:',
                )
            },
        )

        f = build_where_filter(
            schema=schema,
            where_join='AND',
            where={
                'record_id': '12345',
                'date': {'gte': '2025-12-01', 'lte': '2025-12-31'},
            },
        )
        assert f is not None
        clauses = _collect_filter_clauses(f)
        assert {'stringContains': {'key': 'entity_ids', 'value': 'record_id:12345'}} in clauses
        assert {'greaterThanOrEquals': {'key': 'date', 'value': '2025-12-01'}} in clauses
        assert {'lessThanOrEquals': {'key': 'date', 'value': '2025-12-31'}} in clauses

    def test_build_where_filter_or(self):
        """Build an OR-composed where filter across multiple keys."""
        schema = MetadataSchemaFile(
            metadata={
                'entity_ids': MetadataFieldSchema(type='STRING'),
                'actor_ids': MetadataFieldSchema(type='STRING'),
            },
            aliases={
                'record_id': AliasSchema(
                    metadata_key='entity_ids',
                    operator='stringContains',
                    token_prefix='record_id:',
                ),
                'actor_id': AliasSchema(
                    metadata_key='actor_ids',
                    operator='stringContains',
                    token_prefix='actor_id:',
                ),
            },
        )

        f = build_where_filter(
            schema=schema,
            where_join='OR',
            where={'record_id': '12345', 'actor_id': '67890'},
        )
        assert f is not None
        assert 'orAll' in f
        assert {'stringContains': {'key': 'entity_ids', 'value': 'record_id:12345'}} in f['orAll']
        assert {'stringContains': {'key': 'actor_ids', 'value': 'actor_id:67890'}} in f['orAll']

    def test_build_where_filter_missing_key(self):
        """Raise when schema cannot resolve a where key to an alias or metadata key."""
        schema = MetadataSchemaFile(metadata={'other': MetadataFieldSchema(type='STRING')})
        with pytest.raises(FilterError):
            build_where_filter(schema=schema, where={'record_id': '12345'})


class TestRawFilterValidation:
    """Tests for validating raw Bedrock filters against schema."""

    def test_validate_raw_filter_allows_schema_keys(self):
        """Allow valid filters against schema-defined keys."""
        schema = MetadataSchemaFile(metadata={'entity_ids': MetadataFieldSchema(type='STRING')})
        raw_filter = {'stringContains': {'key': 'entity_ids', 'value': 'entity_id:12345'}}
        result = validate_raw_filter_against_schema(schema=schema, raw_filter=raw_filter)
        assert result.ok is True

    def test_validate_raw_filter_rejects_unknown_key(self):
        """Reject filters referencing keys not present in schema."""
        schema = MetadataSchemaFile(metadata={'entity_ids': MetadataFieldSchema(type='STRING')})
        raw_filter = {'equals': {'key': 'unknown', 'value': 'x'}}
        result = validate_raw_filter_against_schema(schema=schema, raw_filter=raw_filter)
        assert result.ok is False
