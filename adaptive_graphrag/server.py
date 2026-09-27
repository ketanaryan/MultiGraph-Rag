import os
import io
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.workflow.state_machine import execute_graphrag_pipeline
from src.storage.pdf_parser import ingest_pdf_documents

app = FastAPI(title="Adaptive GraphRAG API", version="1.0")

# Allow CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, allow all. In prod, lock this down!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for global extractions (simulating a DB state for this prototype)
GLOBAL_STATE = {
    "numerical_extractions": {"invoices": {}, "total_sum": 0.0, "doc_count": 0},
    "uploaded_documents": []
}

class ChatRequest(BaseModel):
    query: str

class MockFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self.content = content
    def read(self):
        return self.content

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    mock_files = []
    for f in files:
        content = await f.read()
        mock_files.append(MockFile(name=f.filename, content=content))
        if f.filename not in GLOBAL_STATE["uploaded_documents"]:
            GLOBAL_STATE["uploaded_documents"].append(f.filename)
            
    # Process documents
    res = ingest_pdf_documents(mock_files)
    
    # Merge numerical extractions state
    GLOBAL_STATE["numerical_extractions"] = res["numerical_extractions"]
    
    return {
        "message": f"Successfully ingested {len(files)} documents",
        "triples_extracted": res["triples_count"],
        "chunks_extracted": res["chunks_count"],
        "uploaded_docs": GLOBAL_STATE["uploaded_documents"]
    }

class FeedbackRequest(BaseModel):
    subgraph: List[Dict[str, Any]]
    is_positive: bool

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    from src.storage.graph_store import get_graph_store
    store = get_graph_store()
    
    # RLHF: Adjust graph temporal weights based on human feedback
    penalty_factor = 0.5 if not request.is_positive else 1.2
    
    modified_count = 0
    for edge in request.subgraph:
        subj = edge.get("subject")
        pred = edge.get("predicate")
        obj = edge.get("object")
        
        # We slash or boost the w0 base weight in the graph store
        for t in store._in_memory_triples:
            if t.get("subject") == subj and t.get("predicate") == pred and t.get("object") == obj:
                current_w0 = float(t.get("w0", 1.0))
                new_w0 = max(0.1, min(1.0, current_w0 * penalty_factor))
                t["w0"] = new_w0
                modified_count += 1
                
    return {
        "status": "success", 
        "message": f"RLHF applied: {modified_count} graph edges adjusted."
    }

import json

MEMORY_FILE = "long_term_memory.json"

def get_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(fact: str):
    m = get_memory()
    if fact not in m:
        m.append(fact)
        with open(MEMORY_FILE, "w") as f:
            json.dump(m, f)

from fastapi.responses import StreamingResponse
import json

@app.post("/api/chat_stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        memories = get_memory()
        
        lower_q = request.query.lower()
        if "i prefer" in lower_q or "always" in lower_q or "remember" in lower_q or "my name" in lower_q:
            save_memory(request.query)
            
        enhanced_query = request.query
        if memories:
            memory_context = " | ".join(memories)
            enhanced_query = f"[LONG-TERM USER MEMORY: {memory_context}]\n\n{request.query}"

        from src.workflow.state_machine import build_graphrag_graph
        app_graph = build_graphrag_graph()
        
        initial_state = {
            "query_raw": enhanced_query,
            "query_rewritten": enhanced_query,
            "complexity_score": 0.0,
            "routing_strategy": "vector",
            "entities": [],
            "estimated_depth": 1,
            "local_density": 1.0,
            "retrieved_vector_chunks": [],
            "retrieved_graph_triples": [],
            "fused_context": "",
            "generated_response": "",
            "verification_score": 0.0,
            "metric_breakdown": {},
            "retry_count": 0,
            "memory_logs": [],
            "execution_trace": [],
            "fallback_invoked": False,
            "numerical_extractions": GLOBAL_STATE.get("numerical_extractions", {}),
            "math_verification": {},
            "uploaded_documents": GLOBAL_STATE.get("uploaded_documents", []),
        }

        final_state = None
        async for output in app_graph.astream(initial_state):
            for node_name, state_update in output.items():
                final_state = state_update
                if node_name == "planner_node":
                    strat = state_update.get("routing_strategy", "vector")
                    yield f'data: {json.dumps({"type": "thought", "agent": "Planner Agent", "message": f"Analyzing query complexity. Selected routing strategy: {strat.upper()}"})}\n\n'
                elif node_name == "retriever_node":
                    triples = state_update.get("retrieved_graph_triples", [])
                    yield f'data: {json.dumps({"type": "thought", "agent": "Retriever Agent", "message": f"Traversed knowledge graph. Found {len(triples)} relevant structural connections."})}\n\n'
                elif node_name == "generator_node":
                    yield f'data: {json.dumps({"type": "thought", "agent": "Generator Agent", "message": "Synthesizing deterministic answer grounded strictly in retrieved graph context..."})}\n\n'
                elif node_name == "verifier_node":
                    score = state_update.get("verification_score", 0.0)
                    if score >= 0.75:
                        yield f'data: {json.dumps({"type": "thought", "agent": "Verifier Agent", "message": f"Mathematical verification passed with score {score:.2f}. Proceeding to output."})}\n\n'
                    else:
                        yield f'data: {json.dumps({"type": "thought", "agent": "Verifier Agent", "message": f"Verification failed (Score: {score:.2f}). Rejecting answer and triggering fallback..."})}\n\n'
                
        if final_state:
            subgraph = final_state.get("retrieved_graph_triples", [])
            # Only keep exact structure to avoid serialization errors
            clean_subgraph = [{"subject": t.get("subject"), "predicate": t.get("predicate"), "object": t.get("object"), "source": t.get("source")} for t in subgraph]
            res_payload = {
                "type": "done",
                "answer": final_state.get("generated_response", "Failed to generate response."),
                "subgraph": clean_subgraph,
                "score": final_state.get("verification_score", 0.0)
            }
            yield f'data: {json.dumps(res_payload)}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
