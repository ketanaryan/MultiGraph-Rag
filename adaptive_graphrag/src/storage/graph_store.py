"""Neo4j graph database interface with temporal edge decay pruning."""

import math
import time
from typing import List, Dict, Any, Optional
from src.config import settings
from src.schema import TripleData


def calculate_temporal_edge_weight(
    w_0: float,
    t_e: float,
    t_0: Optional[float] = None,
    decay_lambda: Optional[float] = None,
) -> float:
    """Calculates temporal edge weight: w(e,t) = w_0 * exp(-lambda * (t_0 - t_e))."""
    current_t = t_0 if t_0 is not None else (time.time() / 86400.0)
    lam = decay_lambda if decay_lambda is not None else settings.temporal_decay_lambda
    delta_t = max(0.0, current_t - t_e)
    weight = w_0 * math.exp(-lam * delta_t)
    return round(float(weight), 4)


class GraphStore:
    """Async-compatible Graph Store interface with temporal edge pruning."""

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> None:
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self._is_connected = False
        self._in_memory_triples: List[Dict[str, Any]] = []
        self._seed_default_graph()

    def _seed_default_graph(self) -> None:
        """Seeds standard test knowledge graph for supply chains and AI chips."""
        current_day = time.time() / 86400.0
        self._in_memory_triples = [
            {"triple_id": "T1", "subject": "TSMC", "predicate": "SUPPLIES_CHIPS_TO", "object": "Apple", "t_e": current_day - 5, "w0": 1.0},
            {"triple_id": "T2", "subject": "TSMC", "predicate": "SUPPLIES_CHIPS_TO", "object": "Nvidia", "t_e": current_day - 12, "w0": 0.95},
            {"triple_id": "T3", "subject": "ASML", "predicate": "PROVIDES_EUV_LITHOGRAPHY_TO", "object": "TSMC", "t_e": current_day - 30, "w0": 1.0},
            {"triple_id": "T4", "subject": "Qualcomm", "predicate": "LICENSES_ARM_ARCHITECTURE_FROM", "object": "Arm", "t_e": current_day - 20, "w0": 0.9},
            {"triple_id": "T5", "subject": "TSMC", "predicate": "MANUFACTURES_SNAPDRAGON_FOR", "object": "Qualcomm", "t_e": current_day - 18, "w0": 0.95},
            {"triple_id": "T6", "subject": "Intel", "predicate": "LEGACY_FOUNDRY_CONTRACT_WITH", "object": "Apple", "t_e": current_day - 300, "w0": 0.8},
            {"triple_id": "T7", "subject": "Nvidia", "predicate": "DEPLOYS_H100_IN", "object": "Microsoft_Azure", "t_e": current_day - 10, "w0": 1.0},
            {"triple_id": "T8", "subject": "Microsoft_Azure", "predicate": "HOSTS_LLM_INFRASTRUCTURE_FOR", "object": "OpenAI", "t_e": current_day - 8, "w0": 1.0},
        ]

    def add_triples(self, new_triples: List[Dict[str, Any]]) -> None:
        """Appends newly ingested triples (e.g. from parsed PDFs) into active graph."""
        existing_ids = {t.get("triple_id") for t in self._in_memory_triples}
        for t in new_triples:
            if t.get("triple_id") not in existing_ids:
                self._in_memory_triples.append(t)
                existing_ids.add(t.get("triple_id"))

    def clear_custom_triples(self) -> None:
        """Resets the in-memory graph to default seed state."""
        self._seed_default_graph()

    async def connect(self) -> bool:
        """Attempts connection to Neo4j driver; gracefully falls back to mock."""
        try:
            from neo4j import AsyncGraphDatabase
            self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
            async with self.driver.session(database=settings.neo4j_database) as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            self._is_connected = True
            return True
        except Exception:
            self._is_connected = False
            return False

    async def close(self) -> None:
        """Closes the active Neo4j driver connection if open."""
        if hasattr(self, "driver") and self.driver:
            await self.driver.close()

    def filter_triples_by_temporal_decay(
        self,
        triples: List[Dict[str, Any]],
        t_0: Optional[float] = None,
        tau_decay: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Applies w(e,t) decay formula and filters out edges below tau_decay (0.3)."""
        threshold = tau_decay if tau_decay is not None else settings.temporal_tau_decay
        surviving_triples: List[Dict[str, Any]] = []

        for item in triples:
            w0 = float(item.get("w0", item.get("weight", 1.0)))
            te = float(item.get("t_e", item.get("timestamp", 0.0)))
            decayed_weight = calculate_temporal_edge_weight(w0, te, t_0)

            if decayed_weight >= threshold:
                item_copy = dict(item)
                item_copy["weight"] = decayed_weight
                surviving_triples.append(item_copy)

        return surviving_triples

    async def query_subgraph(
        self,
        entities: List[str],
        depth: int = 3,
        t_0: Optional[float] = None,
        source_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves subgraphs for entity list traversing up to depth hops (default 3)."""
        if not self._is_connected:
            return self._query_in_memory_subgraph(entities, depth, t_0, source_filter=source_filter)

        cypher = (
            f"MATCH path = (s:Entity)-[r*1..{depth}]-(o:Entity) "
            "WHERE s.name IN $entities "
            "UNWIND relationships(path) as rel "
            "RETURN DISTINCT startNode(rel).name as subject, type(rel) as predicate, "
            "endNode(rel).name as object, rel.timestamp as t_e, "
            "coalesce(rel.weight, 1.0) as w0, id(rel) as triple_id"
        )
        try:
            async with self.driver.session(database=settings.neo4j_database) as session:
                result = await session.run(cypher, parameters={"entities": entities})
                records = [record.data() async for record in result]
                return self.filter_triples_by_temporal_decay(records, t_0)
        except Exception:
            return self._query_in_memory_subgraph(entities, depth, t_0, source_filter=source_filter)

    def _query_in_memory_subgraph(
        self,
        entities: List[str],
        depth: int = 3,
        t_0: Optional[float] = None,
        source_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """In-memory multi-hop graph search traversing up to depth hops (default 3)."""
        candidates = self._in_memory_triples
        if source_filter:
            doc_triples = [
                t for t in self._in_memory_triples
                if t.get("source") in source_filter
                or any(sf.lower() in str(t.get("source", "")).lower() for sf in source_filter)
            ]
            if doc_triples:
                candidates = doc_triples
            elif not self._in_memory_triples:
                return []

        if not entities:
            return self.filter_triples_by_temporal_decay(candidates, t_0)

        matched_triples: List[Dict[str, Any]] = []
        visited_ids = set()
        normalized_entities = [e.lower().strip() for e in entities if e.strip()]
        current_frontier = set(normalized_entities)

        for _ in range(max(1, depth)):
            next_frontier = set()
            added_in_hop = False
            for item in candidates:
                tid = item.get("triple_id") or f"{item.get('subject')}_{item.get('predicate')}_{item.get('object')}"
                if tid in visited_ids:
                    continue
                s_lower = str(item.get("subject", "")).lower()
                o_lower = str(item.get("object", "")).lower()
                matches = any(
                    ent in s_lower or s_lower in ent or ent in o_lower or o_lower in ent
                    for ent in current_frontier
                )
                if matches:
                    matched_triples.append(item)
                    visited_ids.add(tid)
                    next_frontier.add(s_lower)
                    next_frontier.add(o_lower)
                    added_in_hop = True
            if not added_in_hop or not next_frontier:
                break
            current_frontier = next_frontier

        if not matched_triples and source_filter:
            matched_triples = candidates

        fallback_triples = candidates if source_filter else self._in_memory_triples[:8]
        return self.filter_triples_by_temporal_decay(matched_triples or fallback_triples, t_0)


_graph_store_instance: Optional[GraphStore] = None


def get_graph_store() -> GraphStore:
    """Singleton getter for GraphStore."""
    global _graph_store_instance
    if _graph_store_instance is None:
        _graph_store_instance = GraphStore()
    return _graph_store_instance
