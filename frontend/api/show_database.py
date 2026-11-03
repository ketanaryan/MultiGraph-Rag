import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.storage.graph_store import get_graph_store
from src.storage.vector_store import get_vector_store

def main():
    print("=" * 70)
    print("      MULTI-AGENT GRAPHRAG: ACTIVE DATABASE STATUS & INSPECTOR")
    print("=" * 70)
    
    g_store = get_graph_store()
    v_store = get_vector_store()
    
    print("\n[1] GRAPH DATABASE (Neo4j Architecture / Property Graph):")
    print(f"    - Storage Engine: Property Graph with Index-Free Adjacency")
    print(f"    - Connection Mode: {'Remote Neo4j Bolt' if g_store._is_connected else 'In-Memory Zero-Latency Adjacency Store'}")
    print(f"    - Total Active Triples (Edges): {len(g_store._in_memory_triples)}")
    
    if g_store._in_memory_triples:
        print("\n    Sample Knowledge Graph Triples in Database:")
        for idx, t in enumerate(g_store._in_memory_triples[:10], start=1):
            s = t.get("subject")
            r = t.get("relation") or t.get("predicate")
            o = t.get("object")
            w = t.get("weight") or t.get("w0", 1.0)
            src = t.get("source", "PDF")
            print(f"      {idx:02d}. ({s}) --[{r}]--> ({o}) | Weight: {w:.2f} | Source: {src}")
    else:
        print("    (Empty / Ready for PDF upload. When you upload PDFs, triples appear here instantly!)")

    print("\n" + "-" * 70)
    print("\n[2] VECTOR DATABASE (ChromaDB / Dense Semantic Vectors):")
    print(f"    - Storage Engine: ChromaDB (HNSW Cosine Similarity Index)")
    print(f"    - Embedding Dimension: 1536-dim Dense Embeddings")
    print(f"    - Total Chunks Indexed: {len(v_store._documents)}")
    
    if v_store._documents:
        print("\n    Sample Vector Chunks in Database:")
        for idx, doc in enumerate(v_store._documents[:5], start=1):
            cid = doc.get("chunk_id", "N/A")
            txt = doc.get("text", "")[:75].replace("\n", " ")
            print(f"      {idx:02d}. [{cid}] {txt}...")
    else:
        print("    (Empty / Ready for PDF upload. When you upload PDFs, text chunks appear here!)")
        
    print("\n" + "=" * 70)
    print("  HOW TO ACCESS VISUALLY:")
    print("  Run 'START_PROJECT_DEMO.bat' -> Press [1] (Streamlit UI)")
    print("  Open browser: http://localhost:8501 -> Click 'Knowledge Subgraph' Tab")
    print("=" * 70)

if __name__ == "__main__":
    main()
