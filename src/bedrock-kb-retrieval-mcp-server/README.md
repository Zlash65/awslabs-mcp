# Amazon Bedrock Knowledge Base Retrieval MCP Server

MCP server for accessing Amazon Bedrock Knowledge Bases

## Features

### Discover knowledge bases and their data sources

- Find and explore all available knowledge bases
- Search for knowledge bases by name or tag
- List data sources associated with each knowledge base

### Query knowledge bases with natural language

- Retrieve information using conversational queries
- Get relevant passages from your knowledge bases
- Access citation information for all results
- Use `QueryKnowledgeBasesWithMetadata` for discovery when you need metadata fields (date, users, companies, IDs)

### Filter results by data source

- Focus your queries on specific data sources
- Include or exclude specific data sources
- Prioritize results from specific data sources

### Rerank results

- Improve relevance of retrieval results
- Use Amazon Bedrock reranking capabilities
- Sort results by relevance to your query

## Prerequisites

### Installation Requirements

1. Install `uv` from [Astral](https://docs.astral.sh/uv/getting-started/installation/) or the [GitHub README](https://github.com/astral-sh/uv#installation)
2. Install Python using `uv python install 3.10`

### AWS Requirements

1. **AWS CLI Configuration**: You must have the AWS CLI configured with credentials and an AWS_PROFILE that has access to Amazon Bedrock and Knowledge Bases
2. **Amazon Bedrock Knowledge Base**: You must have at least one Amazon Bedrock Knowledge Base with the tag key `mcp-multirag-kb` with a value of `true`
3. **IAM Permissions**: Your IAM role/user must have appropriate permissions to:
   - List and describe knowledge bases
   - Access data sources
   - Query knowledge bases

### Reranking Requirements

If you intend to use reranking functionality, your Bedrock Knowledge Base needs additional permissions:

1. Your IAM role must have permissions for both `bedrock:Rerank` and `bedrock:InvokeModel` actions
2. The Amazon Bedrock Knowledge Bases service role must also have these permissions
3. Reranking is only available in specific regions. Please refer to the official [documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/rerank-supported.html) for an up to date list of supported regions.
4. Enable model access for the available reranking models in the specified region.

### Controlling Reranking

Reranking can be globally enabled or disabled using the `BEDROCK_KB_RERANKING_ENABLED` environment variable:

- Set to `false` (default): Disables reranking for all queries unless explicitly enabled
- Set to `true`: Enables reranking for all queries unless explicitly disabled

The environment variable accepts various formats:

- For enabling: 'true', '1', 'yes', or 'on' (case-insensitive)
- For disabling: any other value or not set (default behavior)

This setting provides a global default, while individual API calls can still override it by explicitly setting the `reranking` parameter.

### Controlling Search Type (Hybrid vs Semantic)

You can control the Bedrock Knowledge Base retrieval strategy using the `BEDROCK_KB_SEARCH_TYPE` environment variable:

- `DEFAULT` (default): do not set an override; let Bedrock choose a strategy
- `HYBRID`: combines keyword + semantic search (recommended for OpenSearch Serverless)
- `SEMANTIC`: embeddings-only semantic search

This setting provides a global default, while individual tool calls can override it by explicitly setting the `search_type` parameter on `QueryKnowledgeBases` / `QueryKnowledgeBasesWithMetadata`.

### Filtering (Explicit + Implicit)

This server supports metadata-aware retrieval via:

- **Explicit filtering**: callers can pass schema-driven `where` constraints and the server translates them into Bedrock RetrievalFilter expressions.
- **Raw filter passthrough** (optional): callers can pass a raw Bedrock RetrievalFilter object (env-gated).
- **Implicit filtering** (optional, per-call): callers can enable Bedrock implicit filtering to infer filters from natural language queries, using a model ARN and metadata schema.

#### Schema configuration (recommended)

To make filtering reliable and generic across knowledge bases, configure schema paths via environment variables:

- `BEDROCK_KB_SCHEMA_MAP_JSON=/path/schema-map.json` (optional; maps KB IDs to schema files)
- `BEDROCK_KB_SCHEMA_DEFAULT_PATH=/path/default.schema.json` (optional fallback)

This repo includes example schema files you can copy and customize:

- `metadata-schema.example.json`
- `metadata-schema-map.example.json`

If no schema is configured, you can still use the server without filters. If you enable auto schema discovery (`metadata_schema_mode=auto`), the server can infer a minimal schema by sampling retrieval metadata.

#### Schema-driven `where` (recommended)

When a schema is available (static file or auto discovery), you can pass schema-driven constraints via `where`:

```json
{
  "where": {
    "date": { "gte": "2025-12-01", "lte": "2025-12-31" },
    "entity_id": "12345"
  },
  "where_join": "AND"
}
```

Keys must resolve via the schema (either a schema alias, or a direct metadata key). Use `DescribeMetadataSchema` to inspect the effective schema for a knowledge base.

#### Raw filter passthrough (power users)

Raw filter passthrough is disabled by default. Enable it explicitly:

- `BEDROCK_KB_ALLOW_RAW_FILTER=true`

When enabled, raw filters are validated against the resolved schema for the requested knowledge base.

#### Implicit filtering (per-call)

To enable implicit filtering on a call, set `implicit_filter=true` and provide a model ARN via env:

- `BEDROCK_KB_IMPLICIT_FILTER_MODEL_ARN=<arn>`

#### Debugging

Use `DescribeMetadataSchema` to inspect the effective schema used for a knowledge base (and whether it came from a static file or auto-discovery).

For detailed instructions on setting up knowledge bases, see:

- [Create a knowledge base](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-create.html)
- [Managing permissions for Amazon Bedrock knowledge bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-prereq-permissions-general.html)
- [Permissions for reranking in Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/rerank-prereq.html)

## Installation

| Cursor | VS Code |
|:------:|:-------:|
| [![Install MCP Server](https://cursor.com/deeplink/mcp-install-light.svg)](https://cursor.com/en/install-mcp?name=awslabs.bedrock-kb-retrieval-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGF3c2xhYnMuYmVkcm9jay1rYi1yZXRyaWV2YWwtbWNwLXNlcnZlckBsYXRlc3QiLCJlbnYiOnsiQVdTX1BST0ZJTEUiOiJ5b3VyLXByb2ZpbGUtbmFtZSIsIkFXU19SRUdJT04iOiJ1cy1lYXN0LTEiLCJGQVNUTUNQX0xPR19MRVZFTCI6IkVSUk9SIiwiS0JfSU5DTFVTSU9OX1RBR19LRVkiOiJvcHRpb25hbC10YWcta2V5LXRvLWZpbHRlci1rYnMiLCJCRURST0NLX0tCX1JFUkFOS0lOR19FTkFCTEVEIjoiZmFsc2UifSwiZGlzYWJsZWQiOmZhbHNlLCJhdXRvQXBwcm92ZSI6W119) | [![Install on VS Code](https://img.shields.io/badge/Install_on-VS_Code-FF9900?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=Bedrock%20KB%20Retrieval%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22awslabs.bedrock-kb-retrieval-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-profile-name%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%2C%22KB_INCLUSION_TAG_KEY%22%3A%22optional-tag-key-to-filter-kbs%22%2C%22BEDROCK_KB_RERANKING_ENABLED%22%3A%22false%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |

Configure the MCP server in your MCP client configuration (e.g., for Amazon Q Developer CLI, edit `~/.aws/amazonq/mcp.json`):

```json
{
  "mcpServers": {
    "awslabs.bedrock-kb-retrieval-mcp-server": {
      "command": "uvx",
      "args": ["awslabs.bedrock-kb-retrieval-mcp-server@latest"],
      "env": {
        "AWS_PROFILE": "your-profile-name",
        "AWS_REGION": "us-east-1",
        "FASTMCP_LOG_LEVEL": "ERROR",
        "KB_INCLUSION_TAG_KEY": "optional-tag-key-to-filter-kbs",
        "BEDROCK_KB_RERANKING_ENABLED": "false",
        "BEDROCK_KB_SEARCH_TYPE": "HYBRID",
        "BEDROCK_KB_SCHEMA_MAP_JSON": "/path/to/schema-map.json",
        "BEDROCK_KB_SCHEMA_DEFAULT_PATH": "/path/to/default.schema.json",
        "BEDROCK_KB_ALLOW_RAW_FILTER": "false",
        "BEDROCK_KB_IMPLICIT_FILTER_MODEL_ARN": "arn:aws:bedrock:us-west-2::foundation-model/<model-id>"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```
### Windows Installation

For Windows users, the MCP server configuration format is slightly different:

```json
{
  "mcpServers": {
    "awslabs.bedrock-kb-retrieval-mcp-server": {
      "disabled": false,
      "timeout": 60,
      "type": "stdio",
      "command": "uv",
      "args": [
        "tool",
        "run",
        "--from",
        "awslabs.bedrock-kb-retrieval-mcp-server@latest",
        "awslabs.bedrock-kb-retrieval-mcp-server.exe"
      ],
      "env": {
        "FASTMCP_LOG_LEVEL": "ERROR",
        "AWS_PROFILE": "your-aws-profile",
        "AWS_REGION": "us-east-1"
      }
    }
  }
}
```


or docker after a successful `docker build -t awslabs/bedrock-kb-retrieval-mcp-server .`:

```file
# fictitious `.env` file with AWS temporary credentials
AWS_ACCESS_KEY_ID=ASIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_SESSION_TOKEN=AQoEXAMPLEH4aoAH0gNCAPy...truncated...zrkuWJOgQs8IZZaIv2BXIa2R4Olgk
```

```json
  {
    "mcpServers": {
      "awslabs.bedrock-kb-retrieval-mcp-server": {
        "command": "docker",
        "args": [
          "run",
          "--rm",
          "--interactive",
          "--env",
          "FASTMCP_LOG_LEVEL=ERROR",
          "--env",
          "KB_INCLUSION_TAG_KEY=optional-tag-key-to-filter-kbs",
          "--env",
          "BEDROCK_KB_RERANKING_ENABLED=false",
          "--env",
          "AWS_REGION=us-east-1",
          "--env-file",
          "/full/path/to/file/above/.env",
          "awslabs/bedrock-kb-retrieval-mcp-server:latest"
        ],
        "env": {},
        "disabled": false,
        "autoApprove": []
      }
    }
  }
```

NOTE: Your credentials will need to be kept refreshed from your host

## Limitations

- Results with `IMAGE` content type are not included in the KB query response.
- The `reranking` parameter requires additional permissions, Amazon Bedrock model access, and is only available in specific regions.
