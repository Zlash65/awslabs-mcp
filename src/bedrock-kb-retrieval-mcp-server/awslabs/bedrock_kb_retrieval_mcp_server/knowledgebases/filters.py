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

from __future__ import annotations

from awslabs.bedrock_kb_retrieval_mcp_server.knowledgebases.schema import (
    MetadataSchemaFile,
    resolve_where_key,
)
from dataclasses import dataclass
from typing import Any, Literal, Mapping


FriendlyJoin = Literal['AND', 'OR']
WhereOperator = Literal[
    'equals',
    'notEquals',
    'greaterThan',
    'greaterThanOrEquals',
    'lessThan',
    'lessThanOrEquals',
    'in',
    'notIn',
    'startsWith',
    'stringContains',
    'listContains',
]


class FilterError(ValueError):
    """Raised when friendly/raw filtering inputs cannot be translated safely."""


def _and_all(filters: list[dict[str, Any]]) -> dict[str, Any] | None:
    filters = [f for f in filters if f]
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {'andAll': filters}


def _or_all(filters: list[dict[str, Any]]) -> dict[str, Any] | None:
    filters = [f for f in filters if f]
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {'orAll': filters}


def combine_filters_and(filters: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """Combine multiple filters using AND semantics."""
    return _and_all([f for f in filters if f])


def combine_filters_or(filters: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """Combine multiple filters using OR semantics."""
    return _or_all([f for f in filters if f])


def build_data_source_filter(data_source_ids: list[str]) -> dict[str, Any]:
    """Build a filter that restricts retrieval to specific Bedrock data source IDs."""
    return {'in': {'key': 'x-amz-bedrock-kb-data-source-id', 'value': data_source_ids}}


def _canonicalize_token(value: str | int, token_prefix: str | None) -> str:
    raw = str(value).strip()
    if token_prefix:
        if raw.startswith(token_prefix):
            return raw
        return f'{token_prefix}{raw}'
    return raw


def _normalize_where_operator(op: str) -> WhereOperator:
    normalized = op.strip()
    aliases: Mapping[str, WhereOperator] = {
        'eq': 'equals',
        'neq': 'notEquals',
        'gt': 'greaterThan',
        'gte': 'greaterThanOrEquals',
        'lt': 'lessThan',
        'lte': 'lessThanOrEquals',
        'contains': 'stringContains',
        'startswith': 'startsWith',
        'listcontains': 'listContains',
    }
    lowered = normalized.lower()
    resolved = aliases.get(lowered, normalized)
    allowed: set[str] = {
        'equals',
        'notEquals',
        'greaterThan',
        'greaterThanOrEquals',
        'lessThan',
        'lessThanOrEquals',
        'in',
        'notIn',
        'startsWith',
        'stringContains',
        'listContains',
    }
    if resolved not in allowed:
        raise FilterError(f'Unsupported where operator: {op!r}')
    return resolved  # type: ignore[return-value]


def _build_op_clause(
    *,
    op: WhereOperator,
    metadata_key: str,
    token_prefix: str | None,
    value: Any,
) -> dict[str, Any] | None:
    """Build a single-operator filter clause (or a small OR-group for list values)."""
    if value is None:
        return None

    if op in ('in', 'notIn'):
        if not isinstance(value, list):
            raise FilterError(f'Operator {op!r} requires a list value.')
        values = value
        if token_prefix:
            values = [_canonicalize_token(v, token_prefix) for v in value]
        return {op: {'key': metadata_key, 'value': values}}

    if isinstance(value, list):
        parts: list[dict[str, Any]] = []
        for v in value:
            if token_prefix:
                v = _canonicalize_token(v, token_prefix)
            parts.append({op: {'key': metadata_key, 'value': v}})
        return _or_all(parts)

    if token_prefix:
        value = _canonicalize_token(value, token_prefix)
    return {op: {'key': metadata_key, 'value': value}}


def build_where_filter(
    *,
    schema: MetadataSchemaFile,
    where: dict[str, Any] | None,
    where_join: FriendlyJoin = 'AND',
) -> dict[str, Any] | None:
    """Build a Bedrock retrieval filter from a schema-driven `where` mapping.

    `where` keeps the server generic: it accepts arbitrary keys which are resolved via the provided
    metadata schema. Keys can be:
    - schema aliases (preferred)
    - direct metadata keys present in schema.metadata (fallback)

    Value forms:
    - scalar: use schema default operator
    - list: OR-match any of the values for that key
    - dict with explicit operator override: {"op": "...", "value": ...} or {"operator": "...", "value": ...}
    - dict as operator-map: {"gte": "...", "lte": "..."} (AND-composed within that key)
    """
    if not where:
        return None

    groups: list[dict[str, Any]] = []

    for where_key, spec in where.items():
        resolved = resolve_where_key(schema, where_key)
        if resolved is None:
            raise FilterError(
                f'Unknown where key {where_key!r}. Provide it in the schema as an alias, '
                'or use a direct metadata key from DescribeMetadataSchema.'
            )
        metadata_key, default_operator, token_prefix = resolved

        if isinstance(spec, dict):
            if 'op' in spec or 'operator' in spec:
                op_raw = spec.get('op') or spec.get('operator')
                if not isinstance(op_raw, str):
                    raise FilterError(f'where[{where_key!r}].op must be a string.')
                if 'value' not in spec:
                    raise FilterError(f'where[{where_key!r}] must include a "value" field.')
                op = _normalize_where_operator(op_raw)
                groups.append(
                    _build_op_clause(
                        op=op,
                        metadata_key=metadata_key,
                        token_prefix=token_prefix,
                        value=spec.get('value'),
                    )
                    or {}
                )
                continue

            # Operator-map form, AND-composed within the key.
            parts: list[dict[str, Any]] = []
            for op_key, op_value in spec.items():
                if not isinstance(op_key, str):
                    raise FilterError(f'where[{where_key!r}] contains a non-string operator key.')
                op = _normalize_where_operator(op_key)
                clause = _build_op_clause(
                    op=op, metadata_key=metadata_key, token_prefix=token_prefix, value=op_value
                )
                if clause:
                    parts.append(clause)
            group = _and_all(parts)
            if group:
                groups.append(group)
            continue

        # Scalar or list: apply default operator.
        op = _normalize_where_operator(default_operator)
        clause = _build_op_clause(
            op=op, metadata_key=metadata_key, token_prefix=token_prefix, value=spec
        )
        if clause:
            groups.append(clause)

    if where_join == 'OR':
        built = _or_all(groups)
    else:
        built = _and_all(groups)

    if built is None:
        return None

    validation = validate_raw_filter_against_schema(schema=schema, raw_filter=built)
    if not validation.ok:
        raise FilterError(validation.error or 'Friendly where filter validation failed.')
    return built


@dataclass(frozen=True)
class FilterValidationResult:
    """Result of validating a raw Bedrock RetrievalFilter against schema."""

    ok: bool
    error: str | None = None


def _default_allowed_ops_for_type(field_type: str) -> set[str]:
    if field_type == 'NUMBER':
        return {
            'equals',
            'notEquals',
            'greaterThan',
            'greaterThanOrEquals',
            'lessThan',
            'lessThanOrEquals',
            'in',
            'notIn',
        }
    if field_type == 'BOOLEAN':
        return {'equals', 'notEquals', 'in', 'notIn'}
    if field_type == 'STRING_LIST':
        return {'listContains', 'in', 'notIn', 'equals', 'notEquals'}
    return {
        'equals',
        'notEquals',
        'greaterThan',
        'greaterThanOrEquals',
        'lessThan',
        'lessThanOrEquals',
        'in',
        'notIn',
        'startsWith',
        'stringContains',
        'listContains',
    }


def _walk_filter(node: Any, found: list[tuple[str, str, Any]]) -> None:
    if not isinstance(node, dict):
        return
    if 'andAll' in node:
        for child in node.get('andAll') or []:
            _walk_filter(child, found)
        return
    if 'orAll' in node:
        for child in node.get('orAll') or []:
            _walk_filter(child, found)
        return

    if len(node) != 1:
        found.append(('__invalid__', '__invalid__', node))
        return

    op, payload = next(iter(node.items()))
    if not isinstance(payload, dict) or 'key' not in payload:
        found.append(('__invalid__', '__invalid__', node))
        return
    found.append((op, str(payload.get('key')), payload.get('value')))


def validate_raw_filter_against_schema(
    *,
    schema: MetadataSchemaFile,
    raw_filter: dict[str, Any],
) -> FilterValidationResult:
    """Validate a raw Bedrock retrieval filter against the resolved schema.

    This is intentionally conservative:
    - rejects unknown operators or malformed shapes
    - rejects keys not present in schema.metadata (plus the data source key)
    - checks operator compatibility with metadata field type
    """
    found: list[tuple[str, str, Any]] = []
    _walk_filter(raw_filter, found)
    if any(op == '__invalid__' for op, _, _ in found):
        return FilterValidationResult(ok=False, error='Invalid raw filter structure.')

    allowed_keys = set(schema.metadata.keys()) | {'x-amz-bedrock-kb-data-source-id'}

    for op, key, value in found:
        if op in ('andAll', 'orAll'):
            continue
        if op not in {
            'equals',
            'notEquals',
            'greaterThan',
            'greaterThanOrEquals',
            'lessThan',
            'lessThanOrEquals',
            'in',
            'notIn',
            'startsWith',
            'stringContains',
            'listContains',
        }:
            return FilterValidationResult(ok=False, error=f'Unsupported filter operator: {op}')

        if key not in allowed_keys:
            return FilterValidationResult(
                ok=False,
                error=f'Filter key {key!r} is not allowed by schema (KB metadata keys).',
            )

        if key == 'x-amz-bedrock-kb-data-source-id':
            continue

        field = schema.metadata.get(key)
        if field is None:
            return FilterValidationResult(
                ok=False,
                error=f'Filter key {key!r} not present in schema metadata.',
            )

        allowed_ops = set(field.allowed_operators or []) or _default_allowed_ops_for_type(
            field.type
        )
        if op not in allowed_ops:
            return FilterValidationResult(
                ok=False,
                error=f'Operator {op!r} is not allowed for metadata field {key!r} (type {field.type}).',
            )

        if op in ('in', 'notIn'):
            if not isinstance(value, list):
                return FilterValidationResult(
                    ok=False,
                    error=f'Operator {op!r} requires a list value for key {key!r}.',
                )
            continue

        if field.type == 'NUMBER' and not isinstance(value, (int, float)):
            return FilterValidationResult(
                ok=False,
                error=f'Metadata field {key!r} expects a number value.',
            )
        if field.type == 'BOOLEAN' and not isinstance(value, bool):
            return FilterValidationResult(
                ok=False,
                error=f'Metadata field {key!r} expects a boolean value.',
            )
        if field.type in ('STRING', 'STRING_LIST') and not isinstance(value, str):
            return FilterValidationResult(
                ok=False,
                error=f'Metadata field {key!r} expects a string value for operator {op!r}.',
            )

    return FilterValidationResult(ok=True)
