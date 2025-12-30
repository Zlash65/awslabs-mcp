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

import json
from dataclasses import dataclass
from loguru import logger
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import Any, Callable, Literal, Mapping, Optional, Sequence


MetadataType = Literal['STRING', 'NUMBER', 'BOOLEAN', 'STRING_LIST']
SchemaMode = Literal['static', 'auto']
SchemaSource = Literal['map', 'default', 'auto']


class MetadataFieldSchema(BaseModel):
    """Schema for a metadata field returned by Bedrock retrieval."""

    type: MetadataType
    description: str | None = None
    allowed_operators: list[str] | None = None


class AliasSchema(BaseModel):
    """Maps a friendly filter name to a metadata field + matching rules."""

    metadata_key: str
    operator: Literal['equals', 'stringContains', 'startsWith', 'listContains'] = 'stringContains'
    token_prefix: str | None = None


class MetadataSchemaFile(BaseModel):
    """Top-level metadata schema file.

    This schema is used for:
    - validating raw Bedrock retrieval filters (when enabled)
    - translating friendly filter params into Bedrock retrieval filters
    - providing metadataAttributes to Bedrock implicit filtering
    """

    version: int = 1
    metadata: dict[str, MetadataFieldSchema] = Field(default_factory=dict)
    aliases: dict[str, AliasSchema] = Field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedSchema:
    """A resolved schema plus provenance metadata."""

    schema: MetadataSchemaFile
    source: SchemaSource
    cache_hit: bool


def _read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def load_schema_from_path(path: str) -> MetadataSchemaFile:
    """Load and validate a schema file from disk."""
    try:
        raw = _read_json_file(Path(path))
    except FileNotFoundError as e:
        raise ValueError(f'Schema file not found: {path}') from e
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON schema file: {path}') from e

    try:
        return MetadataSchemaFile.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f'Invalid schema file structure: {path}: {e}') from e


def load_schema_map(path: str) -> dict[str, str]:
    """Load a KB-ID -> schema path map."""
    raw = _read_json_file(Path(path))
    if not isinstance(raw, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in raw.items()
    ):
        raise ValueError('Schema map must be a JSON object mapping KB IDs to schema file paths.')
    return raw


def guess_metadata_type(value: Any) -> MetadataType:
    """Best-effort metadata type inference from sample values."""
    if isinstance(value, bool):
        return 'BOOLEAN'
    if isinstance(value, (int, float)):
        return 'NUMBER'
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return 'STRING_LIST'
    return 'STRING'


def auto_discover_schema_from_results(results: Sequence[Mapping[str, Any]]) -> MetadataSchemaFile:
    """Infer a minimal schema from Bedrock retrieval results metadata.

    This is a fallback for generic users; it does not attempt to infer aliases.
    """
    merged: dict[str, MetadataFieldSchema] = {}
    for result in results:
        metadata = result.get('metadata')
        if not isinstance(metadata, dict):
            continue
        for key, value in metadata.items():
            if key not in merged:
                merged[key] = MetadataFieldSchema(
                    type=guess_metadata_type(value),
                    description=f'Auto-discovered from retrieval metadata. Example: {value!r}',
                )
    return MetadataSchemaFile(metadata=merged)


def resolve_schema_for_kb(
    *,
    knowledge_base_id: str,
    schema_mode: SchemaMode,
    schema_map_json_path: str | None,
    schema_default_path: str | None,
    schema_cache: dict[str, tuple[MetadataSchemaFile, SchemaSource]],
    auto_discover_fn: Optional[Callable[[], MetadataSchemaFile]] = None,
) -> ResolvedSchema:
    """Resolve schema for a KB ID.

    Resolution order (per plan):
    1) schema map (if set and contains KB ID)
    2) default schema path (if set)
    3) auto-discovery (if schema_mode == "auto")
    4) error
    """
    if knowledge_base_id in schema_cache:
        cached_schema, cached_source = schema_cache[knowledge_base_id]
        return ResolvedSchema(schema=cached_schema, source=cached_source, cache_hit=True)

    if schema_map_json_path:
        try:
            schema_map = load_schema_map(schema_map_json_path)
        except Exception as e:
            raise ValueError(f'Failed to load schema map: {e}') from e
        schema_path = schema_map.get(knowledge_base_id)
        if schema_path:
            schema = load_schema_from_path(schema_path)
            schema_cache[knowledge_base_id] = (schema, 'map')
            return ResolvedSchema(schema=schema, source='map', cache_hit=False)

    if schema_default_path:
        schema = load_schema_from_path(schema_default_path)
        schema_cache[knowledge_base_id] = (schema, 'default')
        return ResolvedSchema(schema=schema, source='default', cache_hit=False)

    if schema_mode == 'auto':
        if auto_discover_fn is None:
            raise ValueError(
                'Auto schema discovery is enabled but no auto_discover_fn was provided.'
            )
        schema = auto_discover_fn()
        schema_cache[knowledge_base_id] = (schema, 'auto')
        return ResolvedSchema(schema=schema, source='auto', cache_hit=False)

    raise ValueError(
        'No metadata schema available for this knowledge base. '
        'Set BEDROCK_KB_SCHEMA_MAP_JSON and/or BEDROCK_KB_SCHEMA_DEFAULT_PATH, '
        'or enable auto schema discovery via metadata_schema_mode=auto.'
    )


def schema_to_implicit_metadata_attributes(schema: MetadataSchemaFile) -> list[dict[str, Any]]:
    """Convert schema metadata fields into Bedrock implicit filtering metadataAttributes."""
    attributes: list[dict[str, Any]] = []
    for key, field in schema.metadata.items():
        if key.startswith('x-amz-'):
            continue
        attributes.append(
            {
                'key': key,
                'type': field.type,
                'description': field.description or f'Metadata field {key}',
            }
        )
    return attributes


def log_schema_summary(schema: MetadataSchemaFile, knowledge_base_id: str) -> None:
    """Log a small summary of the resolved schema for debugging."""
    logger.info(
        f'Resolved metadata schema for KB {knowledge_base_id}: '
        f'{len(schema.metadata)} metadata keys, {len(schema.aliases)} aliases'
    )


def resolve_where_key(
    schema: MetadataSchemaFile,
    where_key: str,
) -> tuple[str, str, str | None] | None:
    """Resolve a user-provided where key into a concrete metadata key and default matching rules.

    This keeps the MCP server generic:
    - Primary path: resolve user keys via schema-provided `aliases`.
    - Fallback: allow direct metadata keys present in `schema.metadata`.

    Returns:
        A tuple of (metadata_key, default_operator, token_prefix) or None if not resolvable.
    """
    if where_key in schema.aliases:
        alias = schema.aliases[where_key]
        return (alias.metadata_key, alias.operator, alias.token_prefix)

    field = schema.metadata.get(where_key)
    if field is None:
        return None

    if field.type in ('NUMBER', 'BOOLEAN'):
        return (where_key, 'equals', None)
    if field.type == 'STRING_LIST':
        return (where_key, 'listContains', None)
    return (where_key, 'stringContains', None)
