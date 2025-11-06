# Understanding-Elasticsearch-Hands-on-Guide-to-Text-Search-Vs-Semantic-Search-in-Ecommerce-Data
This repo demonstrates how to implement:
- **Text search** (keyword/inverted index)
- **Semantic search** (vector similarity using embeddings)
- **Hybrid search** (text + semantic together)

All examples use an e-commerce product catalog.

## Prerequisites
- Python 3.9+
- Elasticsearch running at `http://localhost:9200`
- Text Embedding Inference (TEI) or similar server at `http://localhost:8080/embeddings`
  - Model: `BAAI/bge-base-en-v1.5` (dim=768)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
