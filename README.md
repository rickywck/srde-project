# Backlog Synthesizer POC

A Python implementation of the Backlog Synthesizer POC that loads Azure DevOps backlog items and architecture constraints into a vector database for semantic search and backlog generation.

## Overview

This POC implements sections 2.1, 2.2, and 5.2 of the Backlog Synthesizer POC plan:

1. **ADO Backlog Loader (2.1)**: Loads Epics, Features, and User Stories from Azure DevOps into Pinecone
2. **Architecture Loader (2.2)**: Loads architecture constraint documents (MD, DOCX, PDF) into Pinecone with smart chunking
3. **Semantic Search Tests (5.2)**: Tests retrieval of relevant items based on intent embeddings and similarity thresholds

## Features

- Azure DevOps REST API integration for backlog retrieval
- Smart document chunking using RecursiveTokenChunker
- OpenAI embeddings (text-embedding-3-small)
- Pinecone vector database storage
- Similarity threshold filtering (default: 0.7)
- Comprehensive test suite

## Prerequisites

- Python 3.9+
- Azure DevOps account with Personal Access Token (PAT)
- Pinecone account and API key
- OpenAI API key

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy environment template and configure:
```bash
cp .env.example .env
```

3. Edit `.env` with your credentials:
```
ADO_PAT=your_ado_personal_access_token
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
```

4. Update `config.poc.yaml` with your project details:
```yaml
ado:
  organization: your-org
  project: your-project

project:
  name: your-project
```

## Usage

### Load ADO Backlog

Load Epics, Features, and User Stories from Azure DevOps into Pinecone:

```bash
python ado_loader.py --organization your-org --project your-project
```

This will:
- Fetch all Epic, Feature, and User Story work items
- Extract title, description, and acceptance criteria
- Generate embeddings using OpenAI
- Store in Pinecone with metadata (work_item_id, work_item_type, state, parent_id, etc.)

### Load Architecture Documents

Load architecture constraint documents into Pinecone:

```bash
python arch_loader.py --project your-project --path ./docs/architecture
```

Supported formats:
- Markdown (.md)
- Microsoft Word (.docx)
- PDF (.pdf)

This will:
- Recursively find all supported documents in the directory
- Chunk documents intelligently (default: 1000 chars with 200 char overlap)
- Generate embeddings for each chunk
- Store in Pinecone with metadata (file_name, chunk_index, etc.)

Options:
- `--chunk-size`: Chunk size in characters (default: 1000)
- `--chunk-overlap`: Overlap between chunks (default: 200)

### Run Semantic Search Tests

Test retrieval functionality:

```bash
# Run with pytest
pytest test_semantic_search.py -v

# Or run manual test
python test_semantic_search.py
```

Tests include:
- Text embedding generation
- ADO item search with similarity threshold
- Architecture constraint search
- Combined search (both ADO + architecture)
- Intent-based query construction

## Configuration

Edit `config.poc.yaml`:

```yaml
retrieval:
  min_similarity_threshold: 0.7  # Adjust similarity threshold (0.0 to 1.0)

openai:
  embedding_model: text-embedding-3-small  # Change embedding model if needed
```

## Project Structure

```
.
├── ado_loader.py              # ADO Backlog Loader (section 2.1)
├── arch_loader.py             # Architecture Loader (section 2.2)
├── test_semantic_search.py    # Semantic search tests (section 5.2)
├── config.poc.yaml            # Configuration file
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
├── sample_architecture.md     # Sample architecture constraints document
└── README.md                  # This file
```

## How It Works

### ADO Loader

1. Connects to Azure DevOps using REST API with PAT authentication
2. Queries for Epic, Feature, and User Story work items using WIQL
3. Fetches full work item details including relationships
4. Combines title + description + acceptance criteria into single text
5. Generates embeddings using OpenAI text-embedding-3-small
6. Stores vectors in Pinecone with metadata for filtering

### Architecture Loader

1. Scans directory for supported document types (.md, .docx, .pdf)
2. Extracts text content from each document
3. Chunks text using RecursiveTokenChunker with semantic separators
4. Generates embeddings for each chunk
5. Stores vectors in Pinecone with chunk metadata

### Semantic Search

1. Builds intent query from dominant intent + labels + segment text
2. Generates query embedding
3. Searches Pinecone separately for:
   - ADO items (filtered by doc_type='ado_backlog')
   - Architecture constraints (filtered by doc_type='architecture')
4. Applies similarity threshold (default: 0.7) to filter results
5. Returns only relevant matches above threshold

## Example: Sample Architecture Document

A sample architecture constraints document is provided in `sample_architecture.md`. Load it with:

```bash
python arch_loader.py --project your-project --path .
```

Then test retrieval:

```bash
python test_semantic_search.py
```

## Troubleshooting

### API Keys Not Found
Ensure your `.env` file is in the project root and contains all required keys.

### No Results from Search
- Check that data has been loaded (run loaders first)
- Try lowering the similarity threshold in `config.poc.yaml`
- Verify your project name matches in both config and namespace

### Import Errors
Install all dependencies: `pip install -r requirements.txt`

Note: The lint errors for imports are expected until dependencies are installed.

## Next Steps

This POC implements the foundation for the full Backlog Synthesizer pipeline. Future enhancements include:

- Segmentation Agent (section 4)
- Backlog Generation Agent (section 5.3-5.4)
- Tagging Agent (section 6)
- Upload & Run API (section 3)
- Evaluation framework (section 8)

## License

MIT
