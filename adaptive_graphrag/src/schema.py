"""Data models and global state schema definitions for Adaptive Agentic GraphRAG."""

from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TripleData(BaseModel):
    """Knowledge graph triple representation with temporal decay weight."""

    triple_id: str = Field(description="Unique identifier for the triple, e.g., T1")
    subject: str = Field(description="Subject entity")
    predicate: str = Field(description="Relationship type")
    object_: str = Field(description="Target entity or literal", alias="object")
    timestamp: float = Field(description="Epoch timestamp or elapsed days")
    weight: float = Field(default=1.0, description="Temporal edge weight w(e,t)")


class VectorChunk(BaseModel):
    """Dense vector document chunk representation."""

    chunk_id: str = Field(description="Document chunk identifier, e.g., Doc1")
    text: str = Field(description="Extracted textual content")
    similarity_score: float = Field(description="Cosine similarity score [0, 1]")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricBreakdown(BaseModel):
    """Component scores contributing to composite verification score."""

    faithfulness: float = Field(ge=0.0, le=1.0, description="S_faith")
    answer_relevance: float = Field(ge=0.0, le=1.0, description="S_ans_rel")
    temporal_validity: float = Field(ge=0.0, le=1.0, description="S_temp")
    composite_score: float = Field(ge=0.0, le=1.0, description="S_total")


class PlannerOutput(BaseModel):
    """Structured output parsed from Planner Agent analysis."""

    query_rewritten: str = Field(description="Optimized retrieval query")
    entities: List[str] = Field(default_factory=list, description="Extracted entity names")
    estimated_depth: int = Field(ge=1, default=1, description="Estimated reasoning hop depth")
    local_density: float = Field(ge=1.0, default=1.0, description="Local neighborhood density")
    complexity_score: float = Field(ge=0.0, le=1.0, default=0.0, description="Calculated C(Q)")
    routing_strategy: str = Field(default="vector", description="'vector' | 'graph' | 'hybrid'")
    reflection_notes: Optional[str] = Field(default=None, description="Diagnostic notes if retrying")


class VerificationOutput(BaseModel):
    """Structured output from Verifier Agent programmatic evaluation."""

    score_total: float = Field(ge=0.0, le=1.0, description="Composite verification score")
    is_verified: bool = Field(description="True if score_total >= threshold")
    metric_breakdown: MetricBreakdown = Field(description="Sub-metric component scores")
    critique: str = Field(description="Actionable diagnostic critique")


class MemoryLogEntry(BaseModel):
    """Structured entry appended to memory_logs upon verification failure."""

    retry_iteration: int
    timestamp: str
    rejected_response: str
    failure_reason: str
    score_total: float
    metric_breakdown: Dict[str, float]
    suggested_query_modification: str


class TraceEvent(BaseModel):
    """Auditable state transition event."""

    node_name: str
    timestamp: str
    duration_ms: float
    details: Dict[str, Any] = Field(default_factory=dict)


class GraphRAGState(TypedDict):
    """LangGraph global immutable state schema."""

    query_raw: str
    query_rewritten: str
    complexity_score: float
    routing_strategy: str  # "vector" | "graph" | "hybrid"
    entities: List[str]
    estimated_depth: int
    local_density: float
    retrieved_vector_chunks: List[Dict[str, Any]]
    retrieved_graph_triples: List[Dict[str, Any]]
    fused_context: str
    generated_response: str
    verification_score: float
    metric_breakdown: Dict[str, float]
    retry_count: int
    memory_logs: List[str]
    execution_trace: List[Dict[str, Any]]
    fallback_invoked: bool
    numerical_extractions: Dict[str, Any]
    math_verification: Dict[str, Any]
    uploaded_documents: List[str]
