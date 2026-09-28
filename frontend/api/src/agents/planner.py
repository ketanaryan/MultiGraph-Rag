"""Planner Agent responsible for query rewriting, entity extraction, and routing policy."""

import re
from typing import List, Optional
from src.config import settings, PromptTemplates
from src.schema import PlannerOutput
from src.routing import (
    calculate_query_complexity,
    determine_routing_strategy,
    estimate_traversal_depth,
)


class PlannerAgent:
    """Decomposes queries, extracts structural entities, and sets retrieval policy."""

    def __init__(self) -> None:
        self.system_prompt = PromptTemplates.PLANNER_SYSTEM_PROMPT
        self.user_template = PromptTemplates.PLANNER_USER_TEMPLATE

    def _extract_entities_heuristic(self, query: str) -> List[str]:
        """Extracts prominent named entities and domain phrases from query."""
        known_entities = [
            "TSMC", "Apple", "Nvidia", "ASML", "Qualcomm", "Arm",
            "Intel", "Microsoft_Azure", "OpenAI", "Transformer", "Transformers"
        ]
        found = [e for e in known_entities if e.lower() in query.lower()]
        
        # Multi-word capitalized phrases (e.g. "Project Aether", "QuantumTech Foundry")
        for p in re.findall(r"\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)+\b", query):
            if p not in found: found.append(p)

        # Single capitalized words excluding common interrogatives
        stopwords = {"What", "How", "Why", "When", "Which", "Who", "Whom", "Whose", "Find", "Compare", "List", "Does", "Is", "Are", "Can", "Tell"}
        for match in re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", query):
            if match not in found and match not in stopwords:
                found.append(match)
                
        return found or ["General_Topic"]

    def _estimate_graph_density(self, entities: List[str]) -> float:
        """Estimates local graph neighborhood density based on focal entities."""
        high_density_hubs = {"tsmc", "apple", "nvidia", "intel"}
        hub_count = sum(1 for e in entities if e.lower() in high_density_hubs)
        return round(1.0 + (hub_count * 0.8), 2)

    async def plan(
        self,
        query_raw: str,
        retry_count: int = 0,
        memory_logs: Optional[List[str]] = None,
    ) -> PlannerOutput:
        """Generates structured planning and routing output."""
        entities = self._extract_entities_heuristic(query_raw)
        depth = estimate_traversal_depth(query_raw)
        density = self._estimate_graph_density(entities)

        # On reflection retries, expand depth and sharpen rewritten query
        if retry_count > 0 and memory_logs:
            depth = min(4, depth + 1)
            query_rewritten = (
                f"{query_raw} [Focus: {', '.join(entities)} with exact relational dependencies]"
            )
            reflection_notes = f"Adapted plan for retry {retry_count} with expanded depth {depth}."
        else:
            query_rewritten = query_raw
            reflection_notes = None

        complexity = calculate_query_complexity(
            query=query_rewritten,
            entities=entities,
            depth=depth,
            density=density,
        )
        routing_strategy = determine_routing_strategy(complexity)

        # If LLM key is configured, invoke LangChain structured model if available
        if settings.openai_api_key and not settings.openai_api_key.startswith("mock"):
            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.prompts import ChatPromptTemplate
                
                llm = ChatOpenAI(
                    model=settings.llm_model,
                    api_key=settings.openai_api_key,
                    temperature=0.1
                ).with_structured_output(PlannerOutput)
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.system_prompt),
                    ("user", self.user_template),
                ])
                chain = prompt | llm
                output = await chain.ainvoke({
                    "query_raw": query_raw,
                    "retry_count": retry_count,
                    "memory_logs": str(memory_logs or []),
                })
                if isinstance(output, PlannerOutput):
                    return output
            except Exception:
                pass

        return PlannerOutput(
            query_rewritten=query_rewritten,
            entities=entities,
            estimated_depth=depth,
            local_density=density,
            complexity_score=complexity,
            routing_strategy=routing_strategy,
            reflection_notes=reflection_notes,
        )
