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
import json
from loguru import logger
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from mypy_boto3_bedrock_agent_runtime.client import AgentsforBedrockRuntimeClient
    from mypy_boto3_bedrock_agent_runtime.type_defs import (
        KnowledgeBaseRetrievalConfigurationTypeDef,
    )
else:
    AgentsforBedrockRuntimeClient = object
    KnowledgeBaseRetrievalConfigurationTypeDef = object


async def query_knowledge_base(
    query: str,
    knowledge_base_id: str,
    kb_agent_client: AgentsforBedrockRuntimeClient,
    number_of_results: int = 20,
    reranking: bool = False,
    reranking_model_name: Literal['COHERE', 'AMAZON'] = 'AMAZON',
    data_source_ids: list[str] | None = None,
    search_type: Literal['HYBRID', 'SEMANTIC', 'DEFAULT'] = 'DEFAULT',
    include_metadata: bool = False,
    content_max_chars: int | None = None,
    retrieval_filter: dict[str, Any] | None = None,
    implicit_filter_configuration: dict[str, Any] | None = None,
) -> str:
    """# Amazon Bedrock Knowledge Base query tool.

    Args:
        query (str): The query to search the knowledge base with.
        knowledge_base_id (str): The knowledge base ID to query.
        kb_agent_client (AgentsforBedrockRuntimeClient): The Bedrock agent client.
        number_of_results (int): The number of results to return.
        reranking (bool): Whether to rerank the results. Can be globally configured using the BEDROCK_KB_RERANKING_ENABLED environment variable.
        reranking_model_name (Literal['COHERE', 'AMAZON']): The name of the reranking model to use.
        data_source_ids (list[str] | None): The data source IDs to filter the knowledge base by.
        search_type (Literal['HYBRID', 'SEMANTIC', 'DEFAULT']): Search strategy for retrieval. If set to
            'DEFAULT', no override is sent and Bedrock chooses a strategy. If set to 'HYBRID' or 'SEMANTIC',
            the request includes an explicit override.
        include_metadata (bool): If True, include Bedrock-returned metadata alongside each result.
        content_max_chars (int | None): If set, truncate returned TEXT content to this many characters.
        retrieval_filter (dict[str, Any] | None): Optional Bedrock RetrievalFilter object to apply. This is
            combined with `data_source_ids` filtering using an `andAll` operation.
        implicit_filter_configuration (dict[str, Any] | None): Optional Bedrock implicitFilterConfiguration to
            enable implicit metadata filtering using a model ARN and metadata schema.

    ## Warning: You must use the `ListKnowledgeBases` tool to get the knowledge base ID and optionally a data source ID first.

    ## Returns:
    - A string containing the results of the query.
    """
    if reranking and kb_agent_client.meta.region_name not in [
        'us-west-2',
        'us-east-1',
        'ap-northeast-1',
        'ca-central-1',
        'eu-central-1',
    ]:
        raise ValueError(
            f'Reranking is not supported in region {kb_agent_client.meta.region_name}'
        )

    retrieve_request: KnowledgeBaseRetrievalConfigurationTypeDef = {
        'vectorSearchConfiguration': {
            'numberOfResults': number_of_results,
        }
    }

    if search_type in ('HYBRID', 'SEMANTIC'):
        retrieve_request['vectorSearchConfiguration']['overrideSearchType'] = search_type  # type: ignore

    filters: list[dict[str, Any]] = []
    if data_source_ids:
        filters.append(
            {
                'in': {
                    'key': 'x-amz-bedrock-kb-data-source-id',
                    'value': data_source_ids,  # type: ignore
                }
            }
        )
    if retrieval_filter:
        filters.append(retrieval_filter)

    if filters:
        if len(filters) == 1:
            retrieve_request['vectorSearchConfiguration']['filter'] = filters[0]  # type: ignore
        else:
            retrieve_request['vectorSearchConfiguration']['filter'] = {'andAll': filters}  # type: ignore

    if implicit_filter_configuration:
        retrieve_request['vectorSearchConfiguration']['implicitFilterConfiguration'] = (  # type: ignore
            implicit_filter_configuration
        )

    if reranking:
        model_name_mapping = {
            'COHERE': 'cohere.rerank-v3-5:0',
            'AMAZON': 'amazon.rerank-v1:0',
        }
        retrieve_request['vectorSearchConfiguration']['rerankingConfiguration'] = {
            'type': 'BEDROCK_RERANKING_MODEL',
            'bedrockRerankingConfiguration': {
                'modelConfiguration': {
                    'modelArn': f'arn:aws:bedrock:{kb_agent_client.meta.region_name}::foundation-model/{model_name_mapping[reranking_model_name]}'
                },
            },
        }

    response = kb_agent_client.retrieve(
        knowledgeBaseId=knowledge_base_id,
        retrievalQuery={'text': query},
        retrievalConfiguration=retrieve_request,
    )
    results = response['retrievalResults']
    documents: list[dict] = []
    for result in results:
        if result['content'].get('type') == 'IMAGE':
            logger.warning('Images are not supported at this time. Skipping...')
            continue
        else:
            content = result['content']
            if content_max_chars and isinstance(content, dict) and content.get('type') == 'TEXT':
                text = content.get('text') or ''
                if isinstance(text, str) and len(text) > content_max_chars:
                    content = {**content, 'text': text[:content_max_chars] + '…'}

            document = {
                'content': content,
                'location': result.get('location', ''),
                'score': result.get('score', ''),
            }
            if include_metadata:
                document['metadata'] = result.get('metadata', {})
            documents.append(document)

    return '\n\n'.join([json.dumps(document) for document in documents])
