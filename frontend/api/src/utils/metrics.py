"""Extrinsic programmatic evaluation metrics for Adaptive Agentic GraphRAG."""

import re
from typing import List, Dict, Any, Optional
from src.config import settings
from src.schema import MetricBreakdown, VerificationOutput


def calculate_faithfulness_score(response: str, context: str) -> float:
    """Calculates S_faith: ratio of sentence claims grounded in context."""
    if not response or not context:
        return 0.0

    if "insufficient context" in response.lower() or "no relevant context" in response.lower():
        return 1.0

    sentences = [s.strip().lstrip("#*- \t") for s in re.split(r"[.!?\n]", response) if len(s.strip().lstrip("#*- \t")) > 10 and not s.strip().startswith("|")]
    if not sentences:
        return 0.85

    context_lower = context.lower()
    context_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", context_lower))
    supported_claims = 0

    stop_words = {
        "the", "and", "this", "that", "with", "from", "for", "verified", "candidate",
        "profile", "synthesized", "active", "document", "evidence", "uploaded", "source",
        "based", "according", "following", "details", "listed", "intelligence", "backend",
        "backends", "technical", "skills", "experience", "education", "competencies",
        "roles", "excerpts", "provided", "context", "knowledge", "graph", "triplets",
        "triplet", "question", "answer", "covered", "between", "here", "also", "both",
        "summary", "overview", "comparison", "feature", "metric", "takeaways", "table"
    }

    for sentence in sentences:
        words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", sentence.lower())
        meaningful_words = [w for w in words if w not in stop_words]
        if not meaningful_words:
            supported_claims += 1
            continue

        matches = sum(
            1 for w in meaningful_words
            if w in context_lower or any(w in ct or ct in w for ct in context_tokens if len(ct) >= 4)
        )
        overlap_ratio = matches / len(meaningful_words)

        if overlap_ratio >= 0.40:
            supported_claims += 1

    return round(float(supported_claims / max(1, len(sentences))), 4)


def calculate_answer_relevance_score(response: str, query: str) -> float:
    """Calculates S_ans_rel: semantic alignment between answer and query."""
    if not response or not query:
        return 0.0

    stop_words = {"what", "does", "have", "who", "whom", "how", "when", "where", "why", "which", "the", "in", "for", "with", "from", "are", "and"}
    q_all = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", query.lower())
    q_filtered = [w for w in q_all if w not in stop_words]
    q_tokens = set(q_filtered or q_all)
    r_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", response.lower()))
    
    if not q_tokens:
        return 0.8
    
    shared = q_tokens.intersection(r_tokens)
    overlap = len(shared) / len(q_tokens)

    # Reward answering direct relational inquiries
    length_penalty = 0.0 if len(response) > 30 else -0.2
    score = min(1.0, max(0.1, (overlap * 0.7) + 0.35 + length_penalty))
    return round(float(score), 4)


def calculate_temporal_validity_score(retrieved_triples: List[Dict[str, Any]]) -> float:
    """Calculates S_temp: mean normalized edge weight of retrieved context triples."""
    if not retrieved_triples:
        return 0.85  # Default for purely vector retrieved chunks

    weights = [float(item.get("weight", 1.0)) for item in retrieved_triples]
    mean_weight = sum(weights) / len(weights)
    return round(float(min(1.0, max(0.0, mean_weight))), 4)


def verify_mathematical_claims(
    response: str,
    context: str,
    numerical_extractions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministically reconciles dollar amounts and calculations across documents."""
    resp_amounts = [float(a.strip().rstrip(".").replace(",", "")) for a in re.findall(r"\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)", response) if a.strip().rstrip(".").replace(",", "").replace(".", "", 1).isdigit()]
    ctx_amounts = [float(a.strip().rstrip(".").replace(",", "")) for a in re.findall(r"\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)", context) if a.strip().rstrip(".").replace(",", "").replace(".", "", 1).isdigit()]
    
    gt_total = None
    if numerical_extractions and "total_sum" in numerical_extractions and numerical_extractions["total_sum"] > 0:
        gt_total = float(numerical_extractions["total_sum"])
    elif ctx_amounts:
        gt_total = sum(ctx_amounts)

    math_reconciled = True
    discrepancy = 0.0
    if resp_amounts and gt_total is not None and any(kw in response.lower() for kw in ["total", "sum", "aggregate", "combined", "overall"]):
        max_resp = max(resp_amounts)
        if abs(max_resp - gt_total) > 0.01:
            math_reconciled = False
            discrepancy = round(max_resp - gt_total, 2)

    return {
        "math_reconciled": math_reconciled,
        "discrepancy": discrepancy,
        "ground_truth_total": gt_total,
        "response_amounts": resp_amounts,
        "context_amounts": ctx_amounts,
    }


def evaluate_extrinsic_verification(
    response: str,
    context: str,
    query: str,
    retrieved_triples: Optional[List[Dict[str, Any]]] = None,
    threshold: Optional[float] = None,
    numerical_extractions: Optional[Dict[str, Any]] = None,
) -> VerificationOutput:
    """Computes S_total = alpha*S_faith + beta*S_ans_rel + gamma*S_temp with math reconciliation."""
    t_verify = threshold if threshold is not None else settings.verification_threshold
    alpha = settings.weight_faithfulness
    beta = settings.weight_answer_relevance
    gamma = settings.weight_temporal_validity

    s_faith = calculate_faithfulness_score(response, context)
    s_ans_rel = calculate_answer_relevance_score(response, query)
    s_temp = calculate_temporal_validity_score(retrieved_triples or [])

    # Deterministic math reconciliation check
    math_check = verify_mathematical_claims(response, context, numerical_extractions)
    if not math_check["math_reconciled"]:
        s_faith = min(s_faith, 0.40)  # Penalize hallucinated calculation

    s_total = round((alpha * s_faith) + (beta * s_ans_rel) + (gamma * s_temp), 4)
    is_verified = s_total >= t_verify

    critique = "Verification passed."
    if not is_verified:
        critique_parts = []
        if not math_check["math_reconciled"]:
            critique_parts.append(f"Math discrepancy detected: mismatch of ${math_check['discrepancy']:,.2f}.")
        if s_faith < 0.70 and math_check["math_reconciled"]:
            critique_parts.append(f"Low faithfulness ({s_faith:.2f}): claims not fully supported by context.")
        if s_ans_rel < 0.70:
            critique_parts.append(f"Low relevance ({s_ans_rel:.2f}): response drifts from query intent.")
        if s_temp < 0.70:
            critique_parts.append(f"Stale temporal graph edges ({s_temp:.2f}).")
        critique = " | ".join(critique_parts) if critique_parts else f"Score {s_total:.2f} below threshold {t_verify}."

    breakdown = MetricBreakdown(
        faithfulness=s_faith,
        answer_relevance=s_ans_rel,
        temporal_validity=s_temp,
        composite_score=s_total,
    )

    return VerificationOutput(
        score_total=s_total,
        is_verified=is_verified,
        metric_breakdown=breakdown,
        critique=critique,
    )
