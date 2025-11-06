import os
import json
import time
from typing import List, Dict, Any, Optional

import requests
from elasticsearch import Elasticsearch, ApiError

# ==============================
# Config
# ==============================
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
TEI_URL = os.getenv("TEI_URL", "http://localhost:8080")  # TEI must expose /embeddings
INDEX_NAME = os.getenv("INDEX_NAME", "products")
EMBED_DIMS = 768
USE_EMBEDDINGS = True  # set False if you want to skip real embeddings (will index zero vectors)


# ==============================
# Sample E-Commerce Catalog
# ==============================
PRODUCTS: List[Dict[str, Any]] = [
    {
        "id": "p001",
        "name": "Wireless Headphones",
        "description": "Over-ear, active noise cancelling, 30h battery life, Bluetooth 5.3.",
        "category": "Audio",
        "brand": "SonicWave",
        "price": 129.0
    },
    {
        "id": "p002",
        "name": "Bluetooth Earphones",
        "description": "Lightweight in-ear earphones with deep bass and clear mic.",
        "category": "Audio",
        "brand": "AirGroove",
        "price": 49.0
    },
    {
        "id": "p003",
        "name": "Wired Headphones",
        "description": "Budget over-ear headset, 3.5mm jack, comfortable cushions.",
        "category": "Audio",
        "brand": "ClearTone",
        "price": 29.0
    },
    {
        "id": "p004",
        "name": "Headphone Case",
        "description": "Hard shell carrying case for over-ear and on-ear headphones.",
        "category": "Accessories",
        "brand": "CarryPro",
        "price": 19.0
    },
    {
        "id": "p005",
        "name": "Wireless Earbuds Pro",
        "description": "True wireless earbuds with ANC, transparency mode, wireless charging.",
        "category": "Audio",
        "brand": "SonicWave",
        "price": 159.0
    },
    {
        "id": "p006",
        "name": "Gaming Headset",
        "description": "RGB over-ear gaming headset, detachable mic, 7.1 surround.",
        "category": "Gaming",
        "brand": "UltraCore",
        "price": 89.0
    },
    {
        "id": "p007",
        "name": "Noise Cancelling Earphones",
        "description": "In-ear wired earphones with passive noise isolation and crisp mids.",
        "category": "Audio",
        "brand": "ClearTone",
        "price": 39.0
    },
    {
        "id": "p008",
        "name": "Bluetooth Speaker",
        "description": "Portable speaker, water-resistant, powerful bass, 12h playback.",
        "category": "Audio",
        "brand": "AirGroove",
        "price": 69.0
    }
]


# ==============================
# Embedding Helper
# ==============================
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Call TEI /embeddings once with a list of strings.
    Returns list of embedding vectors (list[float]) for each input text.
    """
    if not USE_EMBEDDINGS:
        return [[0.0] * EMBED_DIMS for _ in texts]

    # Clean / guard
    cleaned = []
    for t in texts:
        if isinstance(t, str) and t.strip():
            cleaned.append(t)
        else:
            cleaned.append("")

    resp = requests.post(
        f"{TEI_URL}/embeddings",
        headers={"Content-Type": "application/json"},
        json={"input": cleaned},
        timeout=60
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    out = []
    for d in data:
        emb = d.get("embedding")
        if not emb:
            out.append([0.0] * EMBED_DIMS)
        else:
            out.append(list(map(float, emb)))
    return out


# ==============================
# Index Management
# ==============================
def create_index(es: Elasticsearch) -> None:
    """
    Create the products index with a dense_vector field for semantic search.
    """
    if es.indices.exists(index=INDEX_NAME):
        print(f"ℹ️ Index '{INDEX_NAME}' already exists.")
        return

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "name":        {"type": "text"},
                "description": {"type": "text"},
                "category":    {"type": "keyword"},
                "brand":       {"type": "keyword"},
                "price":       {"type": "float"},
                # Dense vector for semantic search
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBED_DIMS,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }

    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"✅ Created index '{INDEX_NAME}'")


def index_products(es: Elasticsearch, products: List[Dict[str, Any]]) -> None:
    """
    Index sample products with embeddings. We embed (name + description).
    """
    texts = [f"{p['name']} — {p['description']}" for p in products]
    embeddings = get_embeddings(texts)

    for p, emb in zip(products, embeddings):
        doc = {
            "name": p["name"],
            "description": p["description"],
            "category": p["category"],
            "brand": p["brand"],
            "price": p["price"],
            "embedding": emb
        }
        es.index(index=INDEX_NAME, id=p["id"], document=doc)
        print(f"   → Indexed {p['id']}: {p['name']}")
    es.indices.refresh(index=INDEX_NAME)
    print("✅ Finished indexing")


# ==============================
# Search Implementations
# ==============================
def text_search(es: Elasticsearch, query_text: str, size: int = 5) -> List[Dict[str, Any]]:
    """
    Keyword-based text search using multi_match across name & description.
    """
    body = {
        "size": size,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": ["name^3", "description"],
                "fuzziness": "AUTO"
            }
        }
    }
    res = es.search(index=INDEX_NAME, body=body)
    return [
        {
            "id": hit.get("_id"),
            "score": round(hit.get("_score", 0.0), 4),
            **hit.get("_source", {})
        }
        for hit in res["hits"]["hits"]
    ]


def semantic_search(es: Elasticsearch, query_text: str, size: int = 5) -> List[Dict[str, Any]]:
    """
    Vector similarity search using cosineSimilarity over the dense_vector field.
    """
    query_vec = get_embeddings([query_text])[0]
    body = {
        "size": size,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.q, 'embedding') + 1.0",
                    "params": {"q": query_vec}
                }
            }
        }
    }
    res = es.search(index=INDEX_NAME, body=body)
    return [
        {
            "id": hit.get("_id"),
            "score": round(hit.get("_score", 0.0), 4),
            **hit.get("_source", {})
        }
        for hit in res["hits"]["hits"]
    ]


def hybrid_search(es: Elasticsearch, query_text: str, size: int = 5) -> List[Dict[str, Any]]:
    """
    Hybrid = text + semantic. Combines multi_match and script_score in a bool should.
    """
    query_vec = get_embeddings([query_text])[0]
    body = {
        "size": size,
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["name^3", "description"],
                            "fuzziness": "AUTO"
                        }
                    },
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.q, 'embedding') + 1.0",
                                "params": {"q": query_vec}
                            }
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        }
    }
    res = es.search(index=INDEX_NAME, body=body)
    return [
        {
            "id": hit.get("_id"),
            "score": round(hit.get("_score", 0.0), 4),
            **hit.get("_source", {})
        }
        for hit in res["hits"]["hits"]
    ]


# ==============================
# Pretty Printing
# ==============================
def print_results(title: str, items: List[Dict[str, Any]]) -> None:
    print("\n" + title)
    print("-" * len(title))
    for i, it in enumerate(items, 1):
        print(f"{i:>2}. [{it['score']}] {it['name']}  —  {it['brand']}  (${it['price']})")
        # Uncomment to see more:
        # print("    ", it["description"])


# ==============================
# Main
# ==============================
def main():
    es = Elasticsearch(ES_URL)

    # 1) Create index (idempotent)
    create_index(es)

    # 2) Index sample data (idempotent-ish; run once or overwrite by reindexing)
    #    Simple existence check: if docs exist, skip reindex
    try:
        count = es.count(index=INDEX_NAME)["count"]
    except ApiError:
        count = 0

    if count == 0:
        print("Index is empty → indexing products with embeddings...")
        index_products(es, PRODUCTS)
    else:
        print(f"Index already has {count} docs → skipping indexing.")

    # 3) Run demo queries
    q_text = "wireless headphones"
    q_sem = "wireless earphones with noise cancelling"

    text_hits = text_search(es, q_text, size=5)
    print_results(f'Text Search: "{q_text}"', text_hits)

    sem_hits = semantic_search(es, q_sem, size=5)
    print_results(f'Semantic Search: "{q_sem}"', sem_hits)

    hybrid_hits = hybrid_search(es, q_sem, size=5)
    print_results(f'Hybrid Search: "{q_sem}"', hybrid_hits)


if __name__ == "__main__":
    main()
