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
"""awslabs Bedrock Knowledge Base Retrieval MCP Server."""

import boto3
import json
import os
import sys
from awslabs.bedrock_kb_retrieval_mcp_server.knowledgebases.clients import (
    get_bedrock_agent_client,
    get_bedrock_agent_runtime_client,
)
from awslabs.bedrock_kb_retrieval_mcp_server.knowledgebases.discovery import (
    DEFAULT_KNOWLEDGE_BASE_TAG_INCLUSION_KEY,
    discover_knowledge_bases,
)
from awslabs.bedrock_kb_retrieval_mcp_server.knowledgebases.filters import (
    FilterError,
    build_where_filter,
    combine_filters_and,
    validate_raw_filter_against_schema,
)
from awslabs.bedrock_kb_retrieval_mcp_server.knowledgebases.retrieval import (
    query_knowledge_base,
)
from awslabs.bedrock_kb_retrieval_mcp_server.knowledgebases.schema import (
    MetadataSchemaFile,
    ResolvedSchema,
    SchemaMode,
    SchemaSource,
    auto_discover_schema_from_results,
    log_schema_summary,
    resolve_schema_for_kb,
    schema_to_implicit_metadata_attributes,
)
from datetime import datetime, timezone
from loguru import logger
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from pydantic.fields import FieldInfo
from starlette.responses import JSONResponse
from typing import Annotated, Any, List, Literal, Optional


# Remove all default handlers then add our own
logger.remove()
logger.add(sys.stderr, level='INFO')


def _parse_bool_env(var_name: str, default: bool) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in ('true', '1', 'yes', 'on'):
        return True
    if raw in ('false', '0', 'no', 'off'):
        return False
    return default


def _parse_int_env(var_name: str, default: int) -> int:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except Exception:
        return default
    if value <= 0:
        return default
    return value


def _parse_transport_env() -> Literal['stdio', 'sse', 'streamable-http']:
    raw = os.getenv('MCP_TRANSPORT', 'stdio').strip().lower()
    if raw in ('stdio', 'sse', 'streamable-http'):
        return raw  # type: ignore[return-value]
    return 'stdio'


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_json_file(path: str) -> None:
    with open(path, 'r', encoding='utf-8') as f:
        json.load(f)


def _get_boto3_session() -> boto3.Session:
    profile = os.getenv('AWS_PROFILE')
    if profile:
        return boto3.Session(profile_name=profile)
    return boto3.Session()


# MCP HTTP runtime configuration (used when MCP_TRANSPORT=streamable-http)
mcp_transport: Literal['stdio', 'sse', 'streamable-http'] = _parse_transport_env()
mcp_host = os.getenv('MCP_HOST', '127.0.0.1').strip() or '127.0.0.1'
mcp_port = _parse_int_env('PORT', 8000)
mcp_stateless = _parse_bool_env('MCP_STATELESS', True)
logger.info(
    f'MCP runtime: transport={mcp_transport} host={mcp_host} port={mcp_port} '
    f'stateless_http={mcp_stateless}'
)

# Parse default search type environment variable.
kb_search_type_raw = os.getenv('BEDROCK_KB_SEARCH_TYPE', 'DEFAULT')
kb_search_type: Literal['HYBRID', 'SEMANTIC', 'DEFAULT'] = 'DEFAULT'
if kb_search_type_raw is not None:
    kb_search_type_raw = kb_search_type_raw.strip().upper()
    if kb_search_type_raw in ('HYBRID', 'SEMANTIC', 'DEFAULT'):
        kb_search_type = kb_search_type_raw  # type: ignore[assignment]
logger.info(f'Default search type: {kb_search_type} (from BEDROCK_KB_SEARCH_TYPE)')

# Parse filter configuration environment variables
kb_allow_raw_filter_raw = os.getenv('BEDROCK_KB_ALLOW_RAW_FILTER', 'false')
kb_allow_raw_filter = kb_allow_raw_filter_raw.strip().lower() in ('true', '1', 'yes', 'on')
logger.info(
    f'Raw filter passthrough enabled: {kb_allow_raw_filter} (from BEDROCK_KB_ALLOW_RAW_FILTER)'
)

kb_schema_map_json_path = os.getenv('BEDROCK_KB_SCHEMA_MAP_JSON')
kb_schema_default_path = os.getenv('BEDROCK_KB_SCHEMA_DEFAULT_PATH')

kb_implicit_filter_model_arn = os.getenv('BEDROCK_KB_IMPLICIT_FILTER_MODEL_ARN')

kb_filter_mode_raw = os.getenv('BEDROCK_KB_FILTER_MODE', 'explicit_then_implicit')
kb_filter_mode_raw = kb_filter_mode_raw.strip().lower()
kb_filter_mode: Literal['none', 'explicit_only', 'implicit_only', 'explicit_then_implicit'] = (
    'explicit_then_implicit'
)
if kb_filter_mode_raw in ('none', 'explicit_only', 'implicit_only', 'explicit_then_implicit'):
    kb_filter_mode = kb_filter_mode_raw  # type: ignore[assignment]
logger.info(f'Default filter mode: {kb_filter_mode} (from BEDROCK_KB_FILTER_MODE)')

_schema_cache: dict[str, tuple[MetadataSchemaFile, SchemaSource]] = {}


global kb_runtime_client
global kb_agent_mgmt_client

try:
    kb_runtime_client = get_bedrock_agent_runtime_client(
        region_name=os.getenv('AWS_REGION'),
        profile_name=os.getenv('AWS_PROFILE'),
    )
    kb_agent_mgmt_client = get_bedrock_agent_client(
        region_name=os.getenv('AWS_REGION'),
        profile_name=os.getenv('AWS_PROFILE'),
    )
except Exception as e:
    logger.error(f'Error getting bedrock agent client: {e}')
    raise e

kb_inclusion_tag_key = os.getenv('KB_INCLUSION_TAG_KEY', DEFAULT_KNOWLEDGE_BASE_TAG_INCLUSION_KEY)

# Parse reranking enabled environment variable
kb_reranking_enabled_raw = os.getenv('BEDROCK_KB_RERANKING_ENABLED')
kb_reranking_enabled = False  # Default value is now False (off)
if kb_reranking_enabled_raw is not None:
    kb_reranking_enabled_raw = kb_reranking_enabled_raw.strip().lower()
    if kb_reranking_enabled_raw in ('true', '1', 'yes', 'on'):
        kb_reranking_enabled = True
logger.info(
    f'Default reranking enabled: {kb_reranking_enabled} (from BEDROCK_KB_RERANKING_ENABLED)'
)

mcp = FastMCP(
    'awslabs.bedrock-kb-retrieval-mcp-server',
    instructions="""
    The AWS Labs Bedrock Knowledge Bases Retrieval MCP Server provides access to Amazon Bedrock Knowledge Bases for retrieving relevant information through natural language queries.

    ## Usage Workflow:
    1. ALWAYS start by using the ListKnowledgeBases tool to discover available knowledge bases and their data sources
    2. Use QueryKnowledgeBasesWithMetadata for an initial discovery pass when you need metadata (date, users, companies, IDs)
    3. Use QueryKnowledgeBases for content-heavy drill-down after you have identified relevant sources/metadata
    4. You can make multiple calls with different queries or targeting different knowledge bases

    ## Important Notes:
    - Knowledge bases contain structured data from various data sources (documents, websites, databases)
    - Each knowledge base has a unique ID that must be used when querying
    - You can filter by specific data sources within a knowledge base using data_source_ids
    - Always verify that the knowledge base ID exists in the ListKnowledgeBases tool response before querying
    """,
    dependencies=['boto3'],
    host=mcp_host,
    port=mcp_port,
    streamable_http_path='/mcp',
    stateless_http=mcp_stateless,
)


@mcp.custom_route('/health', methods=['GET'])
async def health(_request):
    """Basic health check endpoint."""
    return JSONResponse(
        {
            'status': 'ok',
            'timestamp': _now_iso(),
            'service': 'awslabs.bedrock-kb-retrieval-mcp-server',
        }
    )


@mcp.custom_route('/health/ready', methods=['GET'])
async def health_ready(_request):
    """Readiness endpoint that validates AWS credentials via STS."""
    try:
        schema_map_path = os.getenv('BEDROCK_KB_SCHEMA_MAP_JSON')
        if schema_map_path:
            _check_json_file(schema_map_path)

        schema_default_path = os.getenv('BEDROCK_KB_SCHEMA_DEFAULT_PATH')
        if schema_default_path:
            _check_json_file(schema_default_path)

        region_name = os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION')
        sts = _get_boto3_session().client('sts', region_name=region_name)
        sts.get_caller_identity()
    except Exception as e:
        return JSONResponse(
            {
                'status': 'not_ready',
                'timestamp': _now_iso(),
                'service': 'awslabs.bedrock-kb-retrieval-mcp-server',
                'error': str(e),
            },
            status_code=503,
        )
    return JSONResponse(
        {
            'status': 'ready',
            'timestamp': _now_iso(),
            'service': 'awslabs.bedrock-kb-retrieval-mcp-server',
        }
    )


def _auto_discover_schema(
    *,
    knowledge_base_id: str,
    sample_query: str,
    sample_k: int,
) -> MetadataSchemaFile:
    response = kb_runtime_client.retrieve(
        knowledgeBaseId=knowledge_base_id,
        retrievalQuery={'text': sample_query},
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                'numberOfResults': sample_k,
            }
        },
    )
    results = response.get('retrievalResults') or []
    if not isinstance(results, list):
        results = []
    return auto_discover_schema_from_results(results)


def _resolve_schema(
    *,
    knowledge_base_id: str,
    metadata_schema_mode: SchemaMode,
    metadata_schema_auto_sample_query: str,
    metadata_schema_auto_sample_k: int,
) -> ResolvedSchema:
    resolved = resolve_schema_for_kb(
        knowledge_base_id=knowledge_base_id,
        schema_mode=metadata_schema_mode,
        schema_map_json_path=kb_schema_map_json_path,
        schema_default_path=kb_schema_default_path,
        schema_cache=_schema_cache,
        auto_discover_fn=lambda: _auto_discover_schema(
            knowledge_base_id=knowledge_base_id,
            sample_query=metadata_schema_auto_sample_query,
            sample_k=metadata_schema_auto_sample_k,
        ),
    )
    log_schema_summary(resolved.schema, knowledge_base_id)
    return resolved


def _build_implicit_filter_configuration(
    *,
    schema: MetadataSchemaFile,
) -> dict[str, Any]:
    if not kb_implicit_filter_model_arn:
        raise ValueError(
            'Implicit filtering requires BEDROCK_KB_IMPLICIT_FILTER_MODEL_ARN to be set.'
        )
    metadata_attributes = schema_to_implicit_metadata_attributes(schema)
    if not metadata_attributes:
        raise ValueError('No usable metadata attributes available for implicit filtering.')
    return {
        'metadataAttributes': metadata_attributes,
        'modelArn': kb_implicit_filter_model_arn,
    }


def _unwrap_field_default(value: Any) -> Any:
    """Unwrap pydantic FieldInfo defaults when tool functions are called directly.

    FastMCP uses pydantic Field(...) in function signatures. When these functions are called
    directly (as in unit tests), Python assigns the FieldInfo object as the default value.
    We normalize those to their underlying `.default` values for internal logic.
    """
    if isinstance(value, FieldInfo):
        return value.default
    return value


@mcp.tool(name='DescribeMetadataSchema')
async def describe_metadata_schema_tool(
    knowledge_base_id: str = Field(
        ...,
        description='The knowledge base ID to inspect. It must be a valid ID from the ListKnowledgeBases tool',
    ),
    metadata_schema_mode: Annotated[
        SchemaMode,
        Field(
            description=(
                "Metadata schema resolution mode. 'static' uses only configured schema files; 'auto' "
                'falls back to sampling retrieval metadata to infer a schema.'
            )
        ),
    ] = 'auto',
    metadata_schema_auto_sample_query: str = Field(
        'the',
        description='Query used for auto schema discovery (only when metadata_schema_mode=auto).',
    ),
    metadata_schema_auto_sample_k: int = Field(
        10,
        description='Number of retrieval results to sample for auto schema discovery.',
    ),
) -> str:
    """Describe the effective metadata schema used for filtering/implicit filtering."""
    metadata_schema_auto_sample_query = _unwrap_field_default(metadata_schema_auto_sample_query)
    metadata_schema_auto_sample_k = _unwrap_field_default(metadata_schema_auto_sample_k)
    resolved = _resolve_schema(
        knowledge_base_id=knowledge_base_id,
        metadata_schema_mode=metadata_schema_mode,
        metadata_schema_auto_sample_query=metadata_schema_auto_sample_query,
        metadata_schema_auto_sample_k=metadata_schema_auto_sample_k,
    )
    return json.dumps(
        {
            'knowledge_base_id': knowledge_base_id,
            'schema_source': resolved.source,
            'cache_hit': resolved.cache_hit,
            'schema': resolved.schema.model_dump(),
        }
    )


@mcp.tool(name='ListKnowledgeBases')
async def list_knowledge_bases_tool() -> str:
    """List all available Amazon Bedrock Knowledge Bases and their data sources.

    This tool returns a mapping of knowledge base IDs to their details, including:
    - name: The human-readable name of the knowledge base
    - description: The description of the knowledge base
    - data_sources: A list of data sources within the knowledge base, each with:
      - id: The unique identifier of the data source
      - name: The human-readable name of the data source

    ## Example response structure:
    ```json
    {
        "kb-12345": {
            "name": "Customer Support KB",
            "description": "Knowledge base containing customer support documentation and FAQs",
            "data_sources": [
                {"id": "ds-abc123", "name": "Technical Documentation"},
                {"id": "ds-def456", "name": "FAQs"}
            ]
        },
        "kb-67890": {
            "name": "Product Information KB",
            "description": "Comprehensive product specifications and details",
            "data_sources": [
                {"id": "ds-ghi789", "name": "Product Specifications"}
            ]
        }
    }
    ```

    ## How to use this information:
    1. Extract the knowledge base IDs (like "kb-12345") for use with the QueryKnowledgeBases tool
    2. Note the data source IDs if you want to filter queries to specific data sources
    3. Use the names to determine which knowledge base and data source(s) are most relevant to the user's query
    """
    knowledge_bases = await discover_knowledge_bases(kb_agent_mgmt_client, kb_inclusion_tag_key)
    return json.dumps(knowledge_bases)


@mcp.tool(name='QueryKnowledgeBases')
async def query_knowledge_bases_tool(
    query: str = Field(
        ..., description='A natural language query to search the knowledge base with'
    ),
    knowledge_base_id: str = Field(
        ...,
        description='The knowledge base ID to query. It must be a valid ID from the ListKnowledgeBases tool',
    ),
    number_of_results: int = Field(
        10,
        description='The number of results to return. Use smaller values for focused results and larger values for broader coverage.',
    ),
    reranking: bool = Field(
        kb_reranking_enabled,
        description='Whether to rerank the results. Useful for improving relevance and sorting. Can be globally configured with BEDROCK_KB_RERANKING_ENABLED environment variable.',
    ),
    reranking_model_name: Literal['COHERE', 'AMAZON'] = Field(
        'AMAZON',
        description="The name of the reranking model to use. Options: 'COHERE', 'AMAZON'",
    ),
    data_source_ids: Optional[List[str]] = Field(
        None,
        description='The data source IDs to filter the knowledge base by. It must be a list of valid data source IDs from the ListKnowledgeBases tool',
    ),
    filter_mode: Annotated[
        Literal['none', 'explicit_only', 'implicit_only', 'explicit_then_implicit'],
        Field(
            description=(
                'How to combine explicit filters (`where`/raw) and implicit filtering. '
                "'explicit_then_implicit' applies explicit filters when provided, and also enables implicit filtering "
                'when requested; it composes constraints with AND semantics.'
            )
        ),
    ] = kb_filter_mode,
    where_join: Annotated[
        Literal['AND', 'OR'],
        Field(
            description=(
                'How to combine schema-driven `where` constraints. Default is AND (narrow). '
                'OR is supported within friendly filters only.'
            )
        ),
    ] = 'AND',
    where: Optional[dict[str, Any]] = Field(
        None,
        description=(
            'Optional schema-driven metadata constraints. Keys are resolved via the metadata schema: '
            'prefer schema aliases, or use direct metadata keys. Values can be scalars, lists (OR within that '
            'key), or dicts for operator overrides, e.g. '
            '{"date": {"gte": "2025-12-01", "lte": "2025-12-31"}, "some_alias": "12345"}.'
        ),
    ),
    filter: Optional[dict[str, Any]] = Field(  # noqa: A002
        None,
        description=(
            'Optional raw Bedrock retrieval filter (RetrievalFilter). '
            'This is a power-user escape hatch and is only accepted when BEDROCK_KB_ALLOW_RAW_FILTER=true.'
        ),
    ),
    implicit_filter: bool = Field(
        False,
        description='Enable Bedrock implicit filtering for this call (requires BEDROCK_KB_IMPLICIT_FILTER_MODEL_ARN).',
    ),
    metadata_schema_mode: Annotated[
        SchemaMode,
        Field(
            description=(
                "Metadata schema resolution mode. 'static' uses only configured schema files; 'auto' "
                'falls back to sampling retrieval metadata to infer a schema.'
            )
        ),
    ] = 'auto',
    metadata_schema_auto_sample_query: str = Field(
        'the',
        description='Query used for auto schema discovery (only when metadata_schema_mode=auto).',
    ),
    metadata_schema_auto_sample_k: int = Field(
        10,
        description='Number of retrieval results to sample for auto schema discovery.',
    ),
    search_type: Annotated[
        Literal['HYBRID', 'SEMANTIC', 'DEFAULT'],
        Field(
            description=(
                "Search strategy for retrieval. 'HYBRID' combines keyword + semantic search and is recommended for "
                "OpenSearch Serverless; 'SEMANTIC' uses embeddings only; 'DEFAULT' lets Bedrock decide."
            )
        ),
    ] = kb_search_type,
) -> str:
    """Query an Amazon Bedrock Knowledge Base using natural language.

    ## Usage Requirements
    - You MUST first use the ListKnowledgeBases tool to get valid knowledge base IDs
    - You can query different knowledge bases or make multiple queries to the same knowledge base

    ## Query Tips
    - Use clear, specific natural language queries for best results
    - You can use this tool MULTIPLE TIMES with different queries to gather comprehensive information
    - Break complex questions into multiple focused queries
    - Consider querying for factual information and explanations separately

    ## Tool output format
    The response contains multiple JSON objects (one per line), each representing a retrieved document with:
    - content: The text content of the document
    - location: The source location of the document
    - score: The relevance score of the document


    ## Interpretation Best Practices
    1. Extract and combine key information from multiple results
    2. Consider the source and relevance score when evaluating information
    3. Use follow-up queries to clarify ambiguous or incomplete information
    4. If the response is not relevant, try a different query, knowledge base, and/or data source
    5. After a few attempts, ask the user for clarification or a different query.
    """
    number_of_results = int(_unwrap_field_default(number_of_results))
    reranking = bool(_unwrap_field_default(reranking))
    reranking_model_name = _unwrap_field_default(reranking_model_name)
    data_source_ids = _unwrap_field_default(data_source_ids)
    where_join = _unwrap_field_default(where_join)
    where = _unwrap_field_default(where)
    filter = _unwrap_field_default(filter)  # noqa: A001
    implicit_filter = bool(_unwrap_field_default(implicit_filter))
    metadata_schema_auto_sample_query = _unwrap_field_default(metadata_schema_auto_sample_query)
    metadata_schema_auto_sample_k = _unwrap_field_default(metadata_schema_auto_sample_k)

    needs_schema = where is not None or filter is not None or implicit_filter

    resolved_schema = None
    if needs_schema:
        resolved_schema = _resolve_schema(
            knowledge_base_id=knowledge_base_id,
            metadata_schema_mode=metadata_schema_mode,
            metadata_schema_auto_sample_query=metadata_schema_auto_sample_query,
            metadata_schema_auto_sample_k=metadata_schema_auto_sample_k,
        )

    raw_filter = filter
    if raw_filter is not None:
        if not kb_allow_raw_filter:
            raise ValueError(
                'Raw filter passthrough is disabled. Set BEDROCK_KB_ALLOW_RAW_FILTER=true to enable.'
            )
        if resolved_schema is None:
            raise ValueError(
                'Raw filters require a resolved metadata schema. Provide a schema via env vars or enable auto mode.'
            )
        validation = validate_raw_filter_against_schema(
            schema=resolved_schema.schema,
            raw_filter=raw_filter,
        )
        if not validation.ok:
            raise ValueError(validation.error or 'Raw filter validation failed.')

    where_filter = None
    if resolved_schema is not None:
        try:
            where_filter = build_where_filter(
                schema=resolved_schema.schema,
                where=where,
                where_join=where_join,
            )
        except FilterError as e:
            raise ValueError(str(e)) from e

    explicit_filter = combine_filters_and([where_filter, raw_filter])

    implicit_filter_configuration = None
    if filter_mode in ('implicit_only', 'explicit_then_implicit') and implicit_filter:
        if resolved_schema is None:
            resolved_schema = _resolve_schema(
                knowledge_base_id=knowledge_base_id,
                metadata_schema_mode=metadata_schema_mode,
                metadata_schema_auto_sample_query=metadata_schema_auto_sample_query,
                metadata_schema_auto_sample_k=metadata_schema_auto_sample_k,
            )
        implicit_filter_configuration = _build_implicit_filter_configuration(
            schema=resolved_schema.schema
        )

    retrieval_filter = None
    if filter_mode == 'explicit_only':
        retrieval_filter = explicit_filter
    elif filter_mode == 'implicit_only':
        retrieval_filter = None
    elif filter_mode == 'explicit_then_implicit':
        retrieval_filter = explicit_filter

    return await query_knowledge_base(
        query=query,
        knowledge_base_id=knowledge_base_id,
        kb_agent_client=kb_runtime_client,
        number_of_results=number_of_results,
        reranking=reranking,
        reranking_model_name=reranking_model_name,
        data_source_ids=data_source_ids,
        search_type=search_type,
        include_metadata=False,
        retrieval_filter=retrieval_filter,
        implicit_filter_configuration=implicit_filter_configuration,
    )


@mcp.tool(name='QueryKnowledgeBasesWithMetadata')
async def query_knowledge_bases_with_metadata_tool(
    query: str = Field(
        ..., description='A natural language query to search the knowledge base with'
    ),
    knowledge_base_id: str = Field(
        ...,
        description='The knowledge base ID to query. It must be a valid ID from the ListKnowledgeBases tool',
    ),
    number_of_results: Optional[int] = Field(
        None,
        description=(
            'The number of results to return. Prefer smaller values for metadata-driven discovery. '
            'If omitted and implicit_filter=true, defaults to 25; otherwise defaults to 6.'
        ),
    ),
    reranking: bool = Field(
        kb_reranking_enabled,
        description='Whether to rerank the results. Useful for improving relevance and sorting. Can be globally configured with BEDROCK_KB_RERANKING_ENABLED environment variable.',
    ),
    reranking_model_name: Literal['COHERE', 'AMAZON'] = Field(
        'AMAZON',
        description="The name of the reranking model to use. Options: 'COHERE', 'AMAZON'",
    ),
    data_source_ids: Optional[List[str]] = Field(
        None,
        description='The data source IDs to filter the knowledge base by. It must be a list of valid data source IDs from the ListKnowledgeBases tool',
    ),
    content_max_chars: Optional[int] = Field(
        600,
        description='If set, truncate returned TEXT content to this many characters to reduce tool output size.',
    ),
    filter_mode: Annotated[
        Literal['none', 'explicit_only', 'implicit_only', 'explicit_then_implicit'],
        Field(
            description=(
                'How to combine explicit filters (`where`/raw) and implicit filtering. '
                "'explicit_then_implicit' applies explicit filters when provided, and also enables implicit filtering "
                'when requested; it composes constraints with AND semantics.'
            )
        ),
    ] = kb_filter_mode,
    where_join: Annotated[
        Literal['AND', 'OR'],
        Field(
            description=(
                'How to combine schema-driven `where` constraints. Default is AND (narrow). '
                'OR is supported within friendly filters only.'
            )
        ),
    ] = 'AND',
    where: Optional[dict[str, Any]] = Field(
        None,
        description=(
            'Optional schema-driven metadata constraints. Keys are resolved via the metadata schema: '
            'prefer schema aliases, or use direct metadata keys. Values can be scalars, lists (OR within that '
            'key), or dicts for operator overrides, e.g. '
            '{"date": {"gte": "2025-12-01", "lte": "2025-12-31"}, "some_alias": "12345"}.'
        ),
    ),
    filter: Optional[dict[str, Any]] = Field(  # noqa: A002
        None,
        description=(
            'Optional raw Bedrock retrieval filter (RetrievalFilter). '
            'This is a power-user escape hatch and is only accepted when BEDROCK_KB_ALLOW_RAW_FILTER=true.'
        ),
    ),
    implicit_filter: bool = Field(
        False,
        description='Enable Bedrock implicit filtering for this call (requires BEDROCK_KB_IMPLICIT_FILTER_MODEL_ARN).',
    ),
    metadata_schema_mode: Annotated[
        SchemaMode,
        Field(
            description=(
                "Metadata schema resolution mode. 'static' uses only configured schema files; 'auto' "
                'falls back to sampling retrieval metadata to infer a schema.'
            )
        ),
    ] = 'auto',
    metadata_schema_auto_sample_query: str = Field(
        'the',
        description='Query used for auto schema discovery (only when metadata_schema_mode=auto).',
    ),
    metadata_schema_auto_sample_k: int = Field(
        10,
        description='Number of retrieval results to sample for auto schema discovery.',
    ),
    search_type: Annotated[
        Literal['HYBRID', 'SEMANTIC', 'DEFAULT'],
        Field(
            description=(
                "Search strategy for retrieval. 'HYBRID' combines keyword + semantic search and is recommended for "
                "OpenSearch Serverless; 'SEMANTIC' uses embeddings only; 'DEFAULT' lets Bedrock decide."
            )
        ),
    ] = kb_search_type,
) -> str:
    """Query an Amazon Bedrock Knowledge Base and include metadata in results.

    Use this tool for a discovery pass when you need metadata fields (e.g., date, users, companies, IDs)
    to filter/sort results. After selecting the most relevant sources, use QueryKnowledgeBases for deeper
    content retrieval.

    ## Usage Requirements
    - You MUST first use the ListKnowledgeBases tool to get valid knowledge base IDs
    - You can query different knowledge bases or make multiple queries to the same knowledge base

    ## Tool output format
    The response contains multiple JSON objects (one per line), each representing a retrieved document with:
    - content: The text content of the document (may be truncated by content_max_chars)
    - location: The source location of the document
    - metadata: Metadata returned by Bedrock Agent Runtime (may be empty depending on KB configuration)
    - score: The relevance score of the document
    """
    number_of_results = _unwrap_field_default(number_of_results)
    reranking = bool(_unwrap_field_default(reranking))
    reranking_model_name = _unwrap_field_default(reranking_model_name)
    data_source_ids = _unwrap_field_default(data_source_ids)
    content_max_chars = _unwrap_field_default(content_max_chars)
    where_join = _unwrap_field_default(where_join)
    where = _unwrap_field_default(where)
    filter = _unwrap_field_default(filter)  # noqa: A001
    implicit_filter = bool(_unwrap_field_default(implicit_filter))
    metadata_schema_auto_sample_query = _unwrap_field_default(metadata_schema_auto_sample_query)
    metadata_schema_auto_sample_k = _unwrap_field_default(metadata_schema_auto_sample_k)

    effective_num_results = number_of_results
    if effective_num_results is None:
        effective_num_results = 25 if implicit_filter else 6

    needs_schema = where is not None or filter is not None or implicit_filter

    resolved_schema = None
    if needs_schema:
        resolved_schema = _resolve_schema(
            knowledge_base_id=knowledge_base_id,
            metadata_schema_mode=metadata_schema_mode,
            metadata_schema_auto_sample_query=metadata_schema_auto_sample_query,
            metadata_schema_auto_sample_k=metadata_schema_auto_sample_k,
        )

    raw_filter = filter
    if raw_filter is not None:
        if not kb_allow_raw_filter:
            raise ValueError(
                'Raw filter passthrough is disabled. Set BEDROCK_KB_ALLOW_RAW_FILTER=true to enable.'
            )
        if resolved_schema is None:
            raise ValueError(
                'Raw filters require a resolved metadata schema. Provide a schema via env vars or enable auto mode.'
            )
        validation = validate_raw_filter_against_schema(
            schema=resolved_schema.schema,
            raw_filter=raw_filter,
        )
        if not validation.ok:
            raise ValueError(validation.error or 'Raw filter validation failed.')

    where_filter = None
    if resolved_schema is not None:
        try:
            where_filter = build_where_filter(
                schema=resolved_schema.schema,
                where=where,
                where_join=where_join,
            )
        except FilterError as e:
            raise ValueError(str(e)) from e

    explicit_filter = combine_filters_and([where_filter, raw_filter])

    implicit_filter_configuration = None
    if filter_mode in ('implicit_only', 'explicit_then_implicit') and implicit_filter:
        if resolved_schema is None:
            resolved_schema = _resolve_schema(
                knowledge_base_id=knowledge_base_id,
                metadata_schema_mode=metadata_schema_mode,
                metadata_schema_auto_sample_query=metadata_schema_auto_sample_query,
                metadata_schema_auto_sample_k=metadata_schema_auto_sample_k,
            )
        implicit_filter_configuration = _build_implicit_filter_configuration(
            schema=resolved_schema.schema
        )

    retrieval_filter = None
    if filter_mode == 'explicit_only':
        retrieval_filter = explicit_filter
    elif filter_mode == 'implicit_only':
        retrieval_filter = None
    elif filter_mode == 'explicit_then_implicit':
        retrieval_filter = explicit_filter

    return await query_knowledge_base(
        query=query,
        knowledge_base_id=knowledge_base_id,
        kb_agent_client=kb_runtime_client,
        number_of_results=effective_num_results,
        reranking=reranking,
        reranking_model_name=reranking_model_name,
        data_source_ids=data_source_ids,
        search_type=search_type,
        include_metadata=True,
        content_max_chars=content_max_chars,
        retrieval_filter=retrieval_filter,
        implicit_filter_configuration=implicit_filter_configuration,
    )


def main():
    """Run the MCP server with CLI argument support."""
    if mcp_transport == 'stdio':
        mcp.run()
        return
    mcp.run(transport=mcp_transport)


if __name__ == '__main__':
    main()
