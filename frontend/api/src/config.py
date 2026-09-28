"""Centralized configuration and prompt repository for Adaptive Agentic GraphRAG."""

from typing import Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application and pipeline configuration parameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # LLM Settings
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-3.5-flash-lite")
    openai_api_key: str = Field(default="mock-key-for-local-execution")
    llm_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimension: int = Field(default=1536)

    # Storage Settings
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="graphrag_password")
    neo4j_database: str = Field(default="neo4j")

    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8000)
    chroma_persist_directory: str = Field(default="./data/chroma")
    chroma_collection_name: str = Field(default="adaptive_graphrag_chunks")

    # Routing Thresholds
    routing_tau_1: float = Field(default=0.35)
    routing_tau_2: float = Field(default=0.70)
    routing_bias_c: float = Field(default=0.85)

    # Temporal Decay
    temporal_decay_lambda: float = Field(default=0.015)
    temporal_tau_decay: float = Field(default=0.30)
    default_edge_weight: float = Field(default=1.0)

    # Verification Parameters
    verification_threshold: float = Field(default=0.75)
    max_retry_count: int = Field(default=3)
    weight_faithfulness: float = Field(default=0.50)
    weight_answer_relevance: float = Field(default=0.30)
    weight_temporal_validity: float = Field(default=0.20)


# Instantiate global singleton settings
settings = Settings()


class PromptTemplates:
    """Externalized prompts for all agent nodes."""

    PLANNER_SYSTEM_PROMPT: str = (
        "You are the Planner Agent for an Adaptive GraphRAG system. "
        "Analyze the user query and produce a structured plan. "
        "Extract key entities, estimate the logical hop depth (1 for direct lookups, "
        "2+ for relationships), and calculate the local graph density. "
        "Rewrite the query to maximize retrieval precision. "
        "If reflection notes are provided, adapt your strategy accordingly."
    )

    PLANNER_USER_TEMPLATE: str = (
        "User Query: {query_raw}\n"
        "Current Retry Count: {retry_count}\n"
        "Reflection Feedback Logs: {memory_logs}\n"
        "Provide your analysis matching the schema."
    )

    GENERATOR_SYSTEM_PROMPT: str = (
        "You are an expert multi-document reasoning agent. "
        "Analyze ALL retrieved context passages and knowledge graph triples across ALL uploaded documents. "
        "Directly answer the user's question in the FIRST sentence before giving supporting details. "
        "If a question requires chaining facts across multiple documents (e.g. Doc A -> Doc B -> Doc C), "
        "explicitly connect the logical chain and state the final answer clearly. "
        "Do NOT output a list of document titles. Synthesize a direct answer."
    )

    GENERATOR_USER_TEMPLATE: str = (
        "Original Query: {query_raw}\n"
        "Rewritten Query: {query_rewritten}\n"
        "Retrieved Fused Context:\n{fused_context}\n\n"
        "Synthesize a factual, grounded response citing sources."
    )

    VERIFIER_SYSTEM_PROMPT: str = (
        "You are the Verifier Agent for an Adaptive GraphRAG system. "
        "Evaluate the candidate response against the retrieved context and user query. "
        "Verify sentence-level claim factual grounding and answer relevance. "
        "Return structured metric scores and actionable diagnostic feedback."
    )

    FALLBACK_RESPONSE_TEMPLATE: str = (
        "I cannot provide a verified answer based on the available knowledge graph "
        "and vector store evidence. The maximum verification retry threshold ({max_retries}) "
        "was reached with composite verification score {score:.2f} (threshold: {threshold}). "
        "Deterministic fallback triggered to prevent hallucination."
    )
