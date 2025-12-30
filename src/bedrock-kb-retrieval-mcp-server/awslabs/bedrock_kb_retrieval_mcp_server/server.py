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
from awslabs.bedrock_kb_retrieval_mcp_server.knowledgebases.retrieval import (
    query_knowledge_base,
)
from loguru import logger
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from typing import Annotated, List, Literal, Optional


# Remove all default handlers then add our own
logger.remove()
logger.add(sys.stderr, level='INFO')

# Parse default search type environment variable.
kb_search_type_raw = os.getenv('BEDROCK_KB_SEARCH_TYPE', 'DEFAULT')
kb_search_type: Literal['HYBRID', 'SEMANTIC', 'DEFAULT'] = 'DEFAULT'
if kb_search_type_raw is not None:
    kb_search_type_raw = kb_search_type_raw.strip().upper()
    if kb_search_type_raw in ('HYBRID', 'SEMANTIC', 'DEFAULT'):
        kb_search_type = kb_search_type_raw  # type: ignore[assignment]
logger.info(f'Default search type: {kb_search_type} (from BEDROCK_KB_SEARCH_TYPE)')


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
    number_of_results: int = Field(
        6,
        description='The number of results to return. Prefer smaller values for metadata-driven discovery.',
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
    return await query_knowledge_base(
        query=query,
        knowledge_base_id=knowledge_base_id,
        kb_agent_client=kb_runtime_client,
        number_of_results=number_of_results,
        reranking=reranking,
        reranking_model_name=reranking_model_name,
        data_source_ids=data_source_ids,
        search_type=search_type,
        include_metadata=True,
        content_max_chars=content_max_chars,
    )


def main():
    """Run the MCP server with CLI argument support."""
    mcp.run()


if __name__ == '__main__':
    main()
