import os
import sys
import asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.storage.graph_store import get_graph_store
from src.storage.vector_store import get_vector_store
from src.config import settings

async def inspect():
    print("=" * 75)
    print("       MULTI-AGENT GRAPHRAG: LIVE PRODUCTION DATABASE INSPECTOR")
    print("=" * 75)
    
    g_store = get_graph_store()
    v_store = get_vector_store()
    
    # Check live Neo4j connection
    is_cloud = await g_store.connect()
    
    print("\n[1] GRAPH DATABASE (Neo4j Enterprise / Aura Cloud):")
    print(f"    - Storage Protocol: Labeled Property Graph (LPG) with Index-Free Adjacency")
    print(f"    - Connection Mode: {'LIVE NEO4J AURA CLOUD CONNECTED' if is_cloud else 'In-Memory Fallback'}")
    print(f"    - Target Cloud URI: {settings.neo4j_uri}")
    print(f"    - Database Name: {settings.neo4j_database}")
    
    if is_cloud:
        try:
            async with g_store.driver.session(database=settings.neo4j_database) as session:
                r1 = await session.run("MATCH (n) RETURN count(n) as node_count")
                rec1 = await r1.single()
                r2 = await session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rec2 = await r2.single()
                print(f"    - Active Cloud Nodes: {rec1['node_count']}")
                print(f"    - Active Cloud Relationships (Edges): {rec2['rel_count']}")
                
                print("\n    Sample Knowledge Graph Triples in Neo4j Aura Cloud:")
                sample = await session.run("MATCH (s)-[r]->(o) RETURN s.name as sub, type(r) as rel, o.name as obj LIMIT 8")
                idx = 1
                async for row in sample:
                    print(f"      {idx:02d}. ({row['sub']}) --[{row['rel']}]--> ({row['obj']})")
                    idx += 1
        except Exception as e:
            print("    Error reading Neo4j Cloud metrics:", e)
    else:
        print(f"    - In-Memory Triples: {len(g_store._in_memory_triples)}")

    print("\n" + "-" * 75)
    print("\n[2] VECTOR DATABASE (ChromaDB / Dense Semantic Vectors):")
    print(f"    - Storage Engine: ChromaDB (HNSW Cosine Similarity Index)")
    print(f"    - Embedding Dimension: 1536-dim Dense Embeddings")
    print(f"    - Collection Name: {settings.chroma_collection_name}")
    print(f"    - Total Chunks Indexed: {len(v_store._documents)}")
    
    if v_store._documents:
        print("\n    Sample Vector Chunks in ChromaDB:")
        for idx, doc in enumerate(v_store._documents[:4], start=1):
            cid = doc.get("chunk_id", "N/A")
            txt = doc.get("text", "")[:70].replace("\n", " ")
            print(f"      {idx:02d}. [{cid}] {txt}...")
    else:
        print("    (ChromaDB Ephemeral Client Active & Ready for dynamic ingestion)")
        
    print("\n" + "=" * 75)
    print("  HOW TO VIEW IN BROWSER:")
    print("  1. Neo4j Cloud Console: Open https://console.neo4j.io -> Click 'Instance01' -> 'Query' or 'Bloom'")
    print("  2. Project Dashboard: Run 'START_PROJECT_DEMO.bat' -> Press [1] -> Click 'Knowledge Subgraph' Tab")
    print("=" * 75)
    
    if is_cloud:
        await g_store.close()

if __name__ == "__main__":
    asyncio.run(inspect())
