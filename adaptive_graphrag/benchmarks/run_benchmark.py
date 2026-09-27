"""Benchmark evaluation harness comparing Adaptive Agentic GraphRAG vs Standard Vector RAG.

Evaluates datasets modeled after HotpotQA (multi-hop relational reasoning)
and FinanceBench (corporate semiconductor supply chains and dependencies).
"""

import sys
import time
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.vector_store import get_vector_store
from src.agents.generator import GeneratorAgent
from src.workflow.state_machine import execute_graphrag_pipeline
from src.utils.metrics import (
    calculate_faithfulness_score,
    calculate_answer_relevance_score,
    evaluate_extrinsic_verification,
)

# Benchmark Dataset: HotpotQA & FinanceBench styles
BENCHMARK_DATASET: List[Dict[str, Any]] = [
    {
        "id": "HQ-01",
        "benchmark": "HotpotQA",
        "type": "multi-hop",
        "query": "Which company supplies EUV lithography to the foundry that manufactures chips for Apple?",
        "ground_truth_entities": ["ASML", "TSMC", "Apple"],
    },
    {
        "id": "HQ-02",
        "benchmark": "HotpotQA",
        "type": "multi-hop",
        "query": "Trace all supply chain dependencies connecting ASML to TSMC and Apple.",
        "ground_truth_entities": ["ASML", "TSMC", "Apple"],
    },
    {
        "id": "HQ-03",
        "benchmark": "HotpotQA",
        "type": "multi-hop",
        "query": "What connects ARM architectures, Qualcomm Snapdragon designs, and TSMC fabrication?",
        "ground_truth_entities": ["Arm", "Qualcomm", "TSMC"],
    },
    {
        "id": "FB-01",
        "benchmark": "FinanceBench",
        "type": "multi-hop",
        "query": "Identify the infrastructure chain connecting Nvidia GPU deployments to OpenAI via Microsoft Azure.",
        "ground_truth_entities": ["Nvidia", "Microsoft_Azure", "OpenAI"],
    },
    {
        "id": "FB-02",
        "benchmark": "FinanceBench",
        "type": "single-hop",
        "query": "Who manufactures custom silicon including A-series and M-series chips for Apple?",
        "ground_truth_entities": ["TSMC", "Apple"],
    },
    {
        "id": "HQ-04",
        "benchmark": "HotpotQA",
        "type": "single-hop",
        "query": "What is the primary role of an attention head in Transformer architectures?",
        "ground_truth_entities": ["Transformer", "attention head"],
    },
]


async def evaluate_standard_vector_rag(item: Dict[str, Any], vector_store, generator: GeneratorAgent) -> Dict[str, Any]:
    """Evaluates query using traditional dense Vector-only RAG (no graph, no verification)."""
    start_t = time.perf_counter()
    chunks = await vector_store.search(item["query"], top_k=3)
    context_str = "\n".join([f"[{c['chunk_id']}]: {c['text']}" for c in chunks])
    
    response = await generator.generate(
        query_raw=item["query"],
        query_rewritten=item["query"],
        fused_context=context_str,
    )
    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

    s_faith = calculate_faithfulness_score(response, context_str)
    s_ans_rel = calculate_answer_relevance_score(response, item["query"])
    
    # Entity coverage
    covered_entities = sum(1 for e in item["ground_truth_entities"] if e.lower() in response.lower())
    entity_recall = round(covered_entities / len(item["ground_truth_entities"]), 3)

    return {
        "id": item["id"],
        "benchmark": item["benchmark"],
        "type": item["type"],
        "query": item["query"],
        "latency_ms": latency_ms,
        "s_faith": s_faith,
        "s_ans_rel": s_ans_rel,
        "entity_recall": entity_recall,
        "response": response,
    }


async def evaluate_adaptive_graphrag(item: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates query using Adaptive Agentic GraphRAG."""
    start_t = time.perf_counter()
    final_state = await execute_graphrag_pipeline(item["query"])
    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

    response = final_state.get("generated_response", "")
    breakdown = final_state.get("metric_breakdown", {})
    s_faith = breakdown.get("faithfulness", 0.0)
    s_ans_rel = breakdown.get("answer_relevance", 0.0)
    s_temp = breakdown.get("temporal_validity", 0.0)
    s_total = final_state.get("verification_score", 0.0)
    
    covered_entities = sum(1 for e in item["ground_truth_entities"] if e.lower() in response.lower())
    entity_recall = round(covered_entities / len(item["ground_truth_entities"]), 3)

    return {
        "id": item["id"],
        "benchmark": item["benchmark"],
        "type": item["type"],
        "query": item["query"],
        "latency_ms": latency_ms,
        "s_faith": s_faith,
        "s_ans_rel": s_ans_rel,
        "s_temp": s_temp,
        "s_total": s_total,
        "routing_strategy": final_state.get("routing_strategy", "vector"),
        "complexity_score": final_state.get("complexity_score", 0.0),
        "retry_count": final_state.get("retry_count", 0),
        "entity_recall": entity_recall,
        "response": response,
    }


async def run_benchmark_suite():
    """Executes comparative evaluation across benchmark dataset."""
    print("=" * 70)
    print("Running Benchmark Evaluation: HotpotQA & FinanceBench")
    print("Comparing: Standard Vector RAG  vs.  Adaptive Agentic GraphRAG")
    print("=" * 70)

    vector_store = get_vector_store()
    generator = GeneratorAgent()

    vector_results = []
    adaptive_results = []

    for item in BENCHMARK_DATASET:
        print(f"\n[Evaluating {item['id']} ({item['benchmark']} - {item['type']})]: {item['query'][:55]}...")
        v_res = await evaluate_standard_vector_rag(item, vector_store, generator)
        a_res = await evaluate_adaptive_graphrag(item)
        vector_results.append(v_res)
        adaptive_results.append(a_res)

        print(f"  -> Vector RAG:     S_faith={v_res['s_faith']:.2f}, Recall={v_res['entity_recall']:.2f}, Latency={v_res['latency_ms']}ms")
        print(f"  -> Adaptive RAG:   S_faith={a_res['s_faith']:.2f}, Recall={a_res['entity_recall']:.2f}, Latency={a_res['latency_ms']}ms [Modality: {a_res['routing_strategy'].upper()}, Retries: {a_res['retry_count']}]")

    # Aggregate statistics
    def avg(lst, key):
        return round(sum(x[key] for x in lst) / len(lst), 3)

    v_avg_faith = avg(vector_results, "s_faith")
    a_avg_faith = avg(adaptive_results, "s_faith")
    v_avg_recall = avg(vector_results, "entity_recall")
    a_avg_recall = avg(adaptive_results, "entity_recall")
    v_avg_lat = avg(vector_results, "latency_ms")
    a_avg_lat = avg(adaptive_results, "latency_ms")

    # Multi-hop specific
    v_mh = [r for r in vector_results if r["type"] == "multi-hop"]
    a_mh = [r for r in adaptive_results if r["type"] == "multi-hop"]
    v_mh_faith = avg(v_mh, "s_faith")
    a_mh_faith = avg(a_mh, "s_faith")
    v_mh_recall = avg(v_mh, "entity_recall")
    a_mh_recall = avg(a_mh, "entity_recall")

    summary = {
        "overall": {
            "standard_vector_rag": {
                "avg_faithfulness": v_avg_faith,
                "avg_entity_recall": v_avg_recall,
                "avg_latency_ms": v_avg_lat,
            },
            "adaptive_graphrag": {
                "avg_faithfulness": a_avg_faith,
                "avg_entity_recall": a_avg_recall,
                "avg_latency_ms": a_avg_lat,
            },
            "deltas": {
                "delta_faithfulness": round(a_avg_faith - v_avg_faith, 3),
                "delta_entity_recall": round(a_avg_recall - v_avg_recall, 3),
                "delta_latency_ms": round(a_avg_lat - v_avg_lat, 2),
            }
        },
        "multi_hop_specific": {
            "vector_faithfulness": v_mh_faith,
            "adaptive_faithfulness": a_mh_faith,
            "delta_faithfulness": round(a_mh_faith - v_mh_faith, 3),
            "vector_entity_recall": v_mh_recall,
            "adaptive_entity_recall": a_mh_recall,
            "delta_entity_recall": round(a_mh_recall - v_mh_recall, 3),
        },
        "detailed_results": {
            "vector_rag": vector_results,
            "adaptive_graphrag": adaptive_results,
        }
    }

    # Save output artifacts
    benchmarks_dir = PROJECT_ROOT / "benchmarks"
    benchmarks_dir.mkdir(exist_ok=True)
    with open(benchmarks_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("BENCHMARK QUANTITATIVE SUMMARY DELTA REPORT")
    print("=" * 70)
    print(f"Overall Faithfulness (S_faith): {v_avg_faith:.2f} -> {a_avg_faith:.2f} (+{summary['overall']['deltas']['delta_faithfulness']:.2f})")
    print(f"Multi-hop Entity Recall:       {v_mh_recall:.2f} -> {a_mh_recall:.2f} (+{summary['multi_hop_specific']['delta_entity_recall']:.2f})")
    print(f"Multi-hop Faithfulness:        {v_mh_faith:.2f} -> {a_mh_faith:.2f} (+{summary['multi_hop_specific']['delta_faithfulness']:.2f})")
    print(f"Mean Pipeline Latency:         {v_avg_lat:.1f}ms (Vector) vs {a_avg_lat:.1f}ms (Adaptive)")
    print(f"Benchmark results successfully saved to: {benchmarks_dir / 'benchmark_results.json'}")

if __name__ == "__main__":
    asyncio.run(run_benchmark_suite())
