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

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        memories = get_memory()
        
        # Simple RLHF / Mem0 heuristic: if user dictates a preference, save it forever.
        lower_q = request.query.lower()
        if "i prefer" in lower_q or "always" in lower_q or "remember" in lower_q or "my name" in lower_q:
            save_memory(request.query)
            
        enhanced_query = request.query
        if memories:
            memory_context = " | ".join(memories)
            enhanced_query = f"[LONG-TERM USER MEMORY: {memory_context}]\n\n{request.query}"

        res = await execute_graphrag_pipeline(
            query=enhanced_query,
            numerical_extractions=GLOBAL_STATE["numerical_extractions"],
            uploaded_documents=GLOBAL_STATE["uploaded_documents"]
        )
        return {
            "answer": res.get("generated_response", "Failed to generate response."),
            "subgraph": res.get("subgraph", []),
            "score": res.get("verification_score", 0.0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
