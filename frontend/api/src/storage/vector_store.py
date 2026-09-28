"""Vector store client interfacing with ChromaDB and dense embeddings."""

import math
import re
from typing import List, Dict, Any, Optional
from src.config import settings
from src.schema import VectorChunk


def compute_token_vector(text: str, vocab: List[str]) -> List[float]:
    """Generates normalized term frequency vector for local fallback."""
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    vec = [1.0 if word in tokens else 0.0 for word in vocab]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class VectorStore:
    """Async-ready Vector Store wrapper with ChromaDB and in-memory fallback."""

    def __init__(self) -> None:
        self.collection_name = settings.chroma_collection_name
        self._is_connected = False
        self._seed_default_documents()
        self._init_chroma()

    def _seed_default_documents(self) -> None:
        """Seeds standard text passages for benchmark evaluation."""
        self._documents: List[Dict[str, Any]] = [
            {
                "chunk_id": "Doc1",
                "text": "The attention head in a Transformer computes scaled dot-product attention over queries, keys, and values to capture contextual token dependencies.",
                "metadata": {"topic": "Deep Learning", "source": "Attention is All You Need"}
            },
            {
                "chunk_id": "Doc2",
                "text": "TSMC manufactures custom silicon including the A-series and M-series chips for Apple using advanced 3nm and 5nm lithography nodes.",
                "metadata": {"topic": "Semiconductors", "source": "Foundry Industry Report"}
            },
            {
                "chunk_id": "Doc3",
                "text": "ASML produces extreme ultraviolet (EUV) photolithography scanners required by leading foundries like TSMC to etch microscopic transistor pathways.",
                "metadata": {"topic": "Semiconductors", "source": "Lithography Review"}
            },
            {
                "chunk_id": "Doc4",
                "text": "Qualcomm designs Snapdragon mobile system-on-chips using ARM architectures and contracts TSMC as their premier semiconductor fabricator.",
                "metadata": {"topic": "Semiconductors", "source": "Qualcomm Filing"}
            },
            {
                "chunk_id": "Doc5",
                "text": "Reciprocal Rank Fusion (RRF) is an information retrieval technique that merges multiple ranked lists without needing normalized score calibration.",
                "metadata": {"topic": "Information Retrieval", "source": "SIGIR Paper"}
            },
            {
                "chunk_id": "Doc6",
                "text": "Apple's primary revenue stream is hardware product sales led by iPhone devices generating over 50% of total net sales, supplemented by Services revenue including App Store, iCloud, and Apple Pay.",
                "metadata": {"topic": "Financial Performance", "source": "Apple 10-K Filing"}
            }
        ]
        all_words = set()
        for doc in self._documents:
            all_words.update(re.findall(r"\b\w+\b", doc["text"].lower()))
        self._vocab = sorted(list(all_words))

    def _init_chroma(self) -> None:
        """Initializes ChromaDB client if library is present."""
        try:
            import chromadb
            self._client = chromadb.EphemeralClient()
            self._collection = self._client.get_or_create_collection(name=self.collection_name)
            ids = [d["chunk_id"] for d in self._documents]
            texts = [d["text"] for d in self._documents]
            metadatas = [d["metadata"] for d in self._documents]
            self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
            self._is_connected = True
        except Exception:
            self._is_connected = False

    def add_documents(self, new_docs: List[Dict[str, Any]]) -> None:
        """Appends newly ingested document chunks (e.g. from PDFs) into vector store."""
        existing_ids = {d["chunk_id"] for d in self._documents}
        to_add = [d for d in new_docs if d["chunk_id"] not in existing_ids]
        if not to_add:
            return
        self._documents.extend(to_add)
        for doc in to_add:
            self._vocab.extend([w for w in doc["text"].lower().split() if w not in self._vocab])
        if self._is_connected and hasattr(self, "_collection"):
            try:
                ids = [d["chunk_id"] for d in to_add]
                texts = [d["text"] for d in to_add]
                metadatas = [d.get("metadata", {}) for d in to_add]
                self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
            except Exception:
                pass

    def clear_custom_documents(self) -> None:
        """Resets vector documents to default seed state."""
        self._seed_default_documents()
        self._init_chroma()

    async def search(self, query: str, top_k: int = 3, source_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Performs dense vector similarity search with cosine ranking and source filtering."""
        if self._is_connected and hasattr(self, "_collection"):
            try:
                where_clause = None
                if source_filter and len(source_filter) == 1:
                    where_clause = {"source": source_filter[0]}
                elif source_filter and len(source_filter) > 1:
                    where_clause = {"source": {"$in": source_filter}}
                
                results = self._collection.query(query_texts=[query], n_results=top_k, where=where_clause)
                parsed: List[Dict[str, Any]] = []
                if results and "documents" in results and results["documents"]:
                    docs = results["documents"][0]
                    ids = results["ids"][0]
                    distances = results["distances"][0] if "distances" in results and results["distances"] else [0.2] * len(docs)
                    for chunk_id, text, dist in zip(ids, docs, distances):
                        sim = max(0.0, min(1.0, 1.0 - float(dist)))
                        parsed.append({"chunk_id": chunk_id, "text": text, "similarity_score": round(sim, 4)})
                    if parsed:
                        return parsed
            except Exception:
                pass

        return self._search_in_memory(query, top_k, source_filter=source_filter)

    def _search_in_memory(self, query: str, top_k: int = 3, source_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """In-memory cosine similarity fallback using normalized token vectors."""
        candidates = self._documents
        if source_filter:
            filtered = [
                d for d in self._documents
                if d.get("metadata", {}).get("source") in source_filter
                or any(sf.lower() in str(d.get("metadata", {}).get("source", "")).lower() for sf in source_filter)
            ]
            if filtered:
                candidates = filtered
            else:
                return []

        q_vec = compute_token_vector(query, self._vocab)
        scored_docs: List[Dict[str, Any]] = []

        for doc in candidates:
            d_vec = compute_token_vector(doc["text"], self._vocab)
            cosine_sim = sum(q * d for q, d in zip(q_vec, d_vec))
            scored_docs.append({
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "similarity_score": round(max(0.15, float(cosine_sim)), 4),
            })

        scored_docs.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_docs[:top_k]


_vector_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Singleton getter for VectorStore."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
