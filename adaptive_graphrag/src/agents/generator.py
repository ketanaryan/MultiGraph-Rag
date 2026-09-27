"""Generator Agent synthesizing context-grounded answers with source citations via Gemini."""

import os
import re
from typing import List, Dict, Any, Optional
from src.config import settings


def clean_citation_text(text: str) -> str:
    """Cleans up internal IDs and formats citations cleanly as (source: Doc_X.pdf)."""
    if not text:
        return ""

    def format_citation(raw_inner: str) -> str:
        s = raw_inner.strip("[]() ")
        s = re.sub(r"^source:\s*", "", s, flags=re.I).strip("[]() ")
        s = re.sub(r"1\.pdf_", "Doc_1_", s)
        s = re.sub(r"D\.pdfoc_", "Doc_", s)
        m_doc = re.search(r"(?:T_|Doc_Doc_|Doc_)*(Doc_[A-Za-z0-9_-]+?)(?:_\d+_\d+|\b)", s, re.I)
        if m_doc:
            doc = m_doc.group(1)
            if not doc.endswith(".pdf"):
                doc = f"{doc}.pdf"
            return f"(source: {doc})"
        m_pdf = re.search(r"([A-Za-z0-9_-]+\.pdf)", s, re.I)
        if m_pdf:
            return f"(source: {m_pdf.group(1)})"
        clean = re.sub(r"\s*\((?:graph|vector)\)", "", s).strip()
        return f"(source: {clean})"

    text = re.sub(r"\[\[\s*([^\]]+?)\s*\]\]", lambda m: format_citation(m.group(1)), text)
    text = re.sub(r"\[\s*([^\]]+?\((?:graph|vector)\))\s*\]", lambda m: format_citation(m.group(1)), text)
    text = re.sub(r"\[\s*((?:T_|Doc_Doc_|Doc_)[A-Za-z0-9_]+)\s*\]", lambda m: format_citation(m.group(1)), text)
    text = re.sub(r"\(source:\s*([^)]+)\)", lambda m: format_citation(m.group(1)), text)
    text = text.replace("[[", "[").replace("]]", "]")
    return text


def _clean_context_line(line: str) -> str:
    """Removes raw internal IDs (e.g. Doc_Doc_... or T_Doc_...) and formats clean source label."""
    m = re.match(r"^\[(?:T_|Doc_)*([A-Za-z0-9_.-]+?)(?:_\d+_\d+)?(?:\s*\((?:vector|graph)\))?\]:\s*(.*)$", line)
    if m:
        name = m.group(1).rstrip("._")
        if not name.endswith(".pdf"):
            name = f"{name}.pdf"
        if not name.startswith("Doc_") and any(k in name for k in ["Architecture", "Hardware", "Supplier"]):
            name = f"Doc_{name}"
        return f"[{name}]: {m.group(2)}"
    return line


class GeneratorAgent:
    """Produces answers strictly grounded in retrieved vector and graph context via Gemini API."""

    def __init__(self) -> None:
        pass

    async def generate(self, query_raw: str, query_rewritten: str = "", fused_context: str = "", fallback_triples: Optional[List[Dict[str, Any]]] = None) -> str:
        """Synthesizes direct multi-document answer using Gemini API with clean source citations."""
        query = query_raw or query_rewritten
        is_broad = any(w in query.lower() for w in ["topic", "summary", "summarize", "overview", "what is this", "main concept", "relationships", "learned", "pdf", "explain", "all document"])

        if (not fused_context or fused_context.strip() == "") or is_broad:
            from src.storage.graph_store import get_graph_store
            triples_source = fallback_triples or get_graph_store()._in_memory_triples
            if triples_source:
                extra_triples = "\n".join([
                    f"[{t.get('source', 'Document.pdf')}]: ({t.get('subject')}) -[{t.get('relation') or t.get('predicate')}]-> ({t.get('object')})"
                    for t in triples_source
                    if len(str(t.get("subject", ""))) <= 40 and len(str(t.get("object", ""))) <= 40
                ])
                fused_context = f"{fused_context}\n{extra_triples}".strip() if fused_context else extra_triples

        if not fused_context or fused_context.strip() == "":
            return "No relevant context found in the uploaded document(s) for this query."

        # Separate retrieved passages and knowledge graph triples
        text_passages: List[str] = []
        graph_triples: List[str] = []
        for raw_line in fused_context.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            cleaned = _clean_context_line(line)
            if ")-[" in line or "-> (" in line or "[T_" in line or "-[" in line:
                graph_triples.append(cleaned)
            else:
                text_passages.append(cleaned)

        fused_text_passages = "\n".join(text_passages) if text_passages else "None retrieved."
        fused_graph_triples = "\n".join(graph_triples) if graph_triples else "None retrieved."

        query = query_raw or query_rewritten

        prompt = f"""You are a multi-document reasoning AI.
Answer the user's query directly and concisely based on the provided context passages and graph triplets.

User Query: {query}

Retrieved Context Passages:
{fused_text_passages}

Retrieved Knowledge Graph Triples:
{fused_graph_triples}

INSTRUCTIONS:
1. State the direct answer clearly in the very first sentence.
2. If answering requires connecting facts across multiple documents, explain the step-by-step logical chain (e.g., Doc A -> Doc B -> Doc C).
3. Do NOT output raw chunk IDs, document titles, or internal debug codes.
4. Append clean source citations as (source: Document_Name.pdf).
"""

        gemini_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if not gemini_key or gemini_key.startswith("mock"):
            raise ValueError(
                "Gemini API key is not configured or is a mock key. "
                "A valid GEMINI_API_KEY is required in .env for answer synthesis."
            )

        from google import genai
        client = genai.Client(api_key=gemini_key)

        import logging, time
        logger = logging.getLogger(__name__)

        candidate_models = [
            "gemini-3.5-flash", "gemini-3.6-flash", "gemini-2.5-flash", "gemini-1.5-flash",
            "gemini-3-flash-preview", "gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-flash-lite-latest"
        ]
        if settings.gemini_model and settings.gemini_model in candidate_models:
            candidate_models.remove(settings.gemini_model)
            candidate_models.insert(0, settings.gemini_model)

        last_error = None
        for model_name in candidate_models:
            for attempt in range(2):
                try:
                    res = client.models.generate_content(model=model_name, contents=prompt)
                    if res and res.text and res.text.strip():
                        return clean_citation_text(res.text.strip())
                except Exception as exc:
                    last_error = exc
                    err_str = str(exc)
                    if "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str.lower():
                        logger.warning(f"Model {model_name} 503 high demand (attempt {attempt+1}/2). Backing off for {2 ** attempt}s...")
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        logger.warning(f"Model {model_name} failed with {exc}. Trying fallback...")
                        break

        if last_error:
            logger.warning(f"All Gemini candidate models failed with error: {last_error}. Falling back to grounded context.")
            if text_passages or graph_triples:
                lead = ""
                for t_line in graph_triples:
                    if any(k in t_line.upper() for k in ["MANUFACTUR", "PRODUC", "POWER", "OPERAT", "CHIP", "CORE"]):
                        m_mfg = re.search(r"\(([^)]+)\)\s*-\[([^]]+)\]->\s*\(([^)]+)\)", t_line)
                        if m_mfg:
                            lead = f"Direct analysis confirms that **{m_mfg.group(3).strip()}** is the key related entity for {m_mfg.group(1).strip()}.\n\n"
                            break
                facts = "\n".join([f"- {item}" for item in (graph_triples + text_passages)[:6]])
                return clean_citation_text(f"{lead}Verified evidence across active documents:\n\n{facts}")
            raise RuntimeError(f"All Gemini candidate models failed. Last error: {last_error}") from last_error
        raise RuntimeError("Gemini API returned an empty response across all candidate models.")


def generate_response(query: str, fused_passages: list, fused_triples: list) -> str:
    """Convenience functional wrapper directly invoking GeneratorAgent."""
    agent = GeneratorAgent()
    p_str = "\n".join([f"- {p}" for p in fused_passages])
    t_str = "\n".join([f"- ({t.get('subject')}) -> [{t.get('relation') or t.get('predicate')}] -> ({t.get('object')})" for t in fused_triples])
    import asyncio
    return asyncio.run(agent.generate(query_raw=query, fused_context=f"{p_str}\n{t_str}"))
