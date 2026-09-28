"""Verifier Agent performing extrinsic algorithmic verification on candidate answers."""

from typing import List, Dict, Any, Optional
from src.config import settings, PromptTemplates
from src.schema import VerificationOutput
from src.utils.metrics import evaluate_extrinsic_verification


class VerifierAgent:
    """Evaluates candidate response via extrinsic algorithmic metrics."""

    def __init__(self) -> None:
        self.system_prompt = PromptTemplates.VERIFIER_SYSTEM_PROMPT
        self.threshold = settings.verification_threshold

    async def verify(
        self,
        query_raw: str,
        generated_response: str,
        fused_context: str,
        retrieved_triples: Optional[List[Dict[str, Any]]] = None,
        threshold: Optional[float] = None,
        numerical_extractions: Optional[Dict[str, Any]] = None,
    ) -> VerificationOutput:
        """Computes S_total = alpha*S_faith + beta*S_ans_rel + gamma*S_temp with math reconciliation."""
        t_verify = threshold if threshold is not None else self.threshold

        return evaluate_extrinsic_verification(
            response=generated_response,
            context=fused_context,
            query=query_raw,
            retrieved_triples=retrieved_triples,
            threshold=t_verify,
            numerical_extractions=numerical_extractions,
        )
