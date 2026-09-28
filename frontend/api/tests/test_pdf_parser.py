"""Unit tests for Multi-PDF Ingestion and Mathematical Verification."""

import io
import pytest
from src.storage.pdf_parser import (
    create_sample_invoice_bytes,
    extract_text_from_pdf_stream,
    parse_financial_entities,
    ingest_pdf_documents,
)
from src.utils.metrics import verify_mathematical_claims, evaluate_extrinsic_verification


def test_pdf_sample_creation_and_text_extraction():
    """Validates synthetic PDF creation and text extraction using pypdf."""
    pdf_bytes = create_sample_invoice_bytes(
        vendor="Acme Cloud Solutions",
        invoice_id="INV-9901",
        amount=15500.00,
        date_str="2026-09-15",
        items=["Dedicated GPU Instance", "Cloud Storage Cluster"]
    )
    assert len(pdf_bytes) > 100
    
    # Extract metadata/text
    stream = io.BytesIO(pdf_bytes)
    text = extract_text_from_pdf_stream(stream)
    assert isinstance(text, str)


def test_parse_financial_entities():
    """Validates regex parsing of invoice metadata from text."""
    sample_text = (
        "INVOICE\nVendor: Apex Lithography BV\nInvoice Number: INV-7720\nDate: 2026-09-12\n"
        "Items:\n- EUV Optical Module: $35,000.00\n- Installation: $7,500.00\nTotal Amount: $42,500.00"
    )
    parsed = parse_financial_entities(sample_text, "Apex_Invoice.pdf")
    
    assert parsed["invoice_id"] == "INV-7720"
    assert "Apex" in parsed["vendor"]
    assert parsed["date"] == "2026-09-12"
    assert parsed["total_amount"] == 42500.00
    assert len(parsed["amounts_found"]) >= 2


def test_multi_pdf_ingestion_and_triples():
    """Validates simultaneous ingestion of multiple PDF documents."""
    pdf1 = create_sample_invoice_bytes("Vendor Alpha", "INV-101", 10000.0, "2026-09-01", ["Item 1"])
    pdf2 = create_sample_invoice_bytes("Vendor Beta", "INV-102", 25000.0, "2026-09-05", ["Item 2"])

    class MockUpload:
        def __init__(self, name: str, data: bytes):
            self.name = name
            self._data = data
        def read(self):
            return self._data

    uploads = [MockUpload("invoice_1.pdf", pdf1), MockUpload("invoice_2.pdf", pdf2)]
    result = ingest_pdf_documents(uploads)

    assert result["triples_count"] >= 6
    assert result["chunks_count"] == 2
    assert "INV-101" in result["numerical_extractions"]["invoices"]
    assert "INV-102" in result["numerical_extractions"]["invoices"]


def test_math_reconciliation_verification():
    """Validates programmatic arithmetic reconciliation and discrepancy detection."""
    context = "Invoice INV-01 is $10,000.00. Invoice INV-02 is $15,000.00."
    numerical_extractions = {"total_sum": 25000.00}

    # Case 1: Matching math
    resp_correct = "The total combined expenditure across invoices is $25,000.00."
    check_ok = verify_mathematical_claims(resp_correct, context, numerical_extractions)
    assert check_ok["math_reconciled"] is True
    assert check_ok["discrepancy"] == 0.0

    # Case 2: Hallucinated / wrong total
    resp_wrong = "The total combined expenditure across invoices is $99,000.00."
    check_fail = verify_mathematical_claims(resp_wrong, context, numerical_extractions)
    assert check_fail["math_reconciled"] is False
    assert check_fail["discrepancy"] == 74000.00

    # Verification gate should penalize mismatch
    v_out = evaluate_extrinsic_verification(
        response=resp_wrong,
        context=context,
        query="What is the total expenditure?",
        numerical_extractions=numerical_extractions,
    )
    assert "Math discrepancy detected" in v_out.critique


def test_resume_skills_extraction_and_triples():
    """Validates parsing of resume text into technical skills and candidate triples."""
    resume_text = (
        "Ashirwad Sharma\nSenior AI Systems Engineer\n"
        "Education: B.Tech Computer Science\n"
        "Experience: Software Engineer at Google\n"
        "Skills: Python, TypeScript, React, Docker, Kubernetes, Neo4j, LangChain, GraphRAG, PyTorch"
    )
    parsed = parse_financial_entities(resume_text, "Ashirwad_Resume.pdf")
    assert parsed["candidate_name"] == "Ashirwad Sharma"
    assert "Python" in parsed["skills"]
    assert "Neo4j" in parsed["skills"]
    assert "GraphRAG" in parsed["skills"]
    assert len(parsed["skills"]) >= 5

    from src.storage.pdf_parser import generate_triples_and_chunks
    triples, chunks = generate_triples_and_chunks(parsed, 1)
    skill_triples = [t for t in triples if t.get("predicate") == "HAS_SKILL"]
    assert len(skill_triples) >= 5
    assert any(t.get("object") == "Python" for t in skill_triples)
    assert any(t.get("object") == "Neo4j" for t in skill_triples)
    assert chunks[0]["metadata"]["source"] == "Ashirwad_Resume.pdf"


def test_fee_challan_dynamic_extraction_and_triples():
    """Validates dynamic extraction of noun phrases and triples from college fee challans."""
    challan_text = (
        "National Institute of Technology\n"
        "Facility Fee Challan\n"
        "Student Name: Rahul Verma\n"
        "Standing: Fourth Year Student\n"
        "Department: Computer Science\n"
        "Fourth Year Student paid Facility Fee Challan\n"
        "Total Amount: Rs. 30,000\n"
        "Date: 2026-09-15"
    )
    parsed = parse_financial_entities(challan_text, "College_Challan.pdf")
    assert parsed["candidate_name"] == "Rahul Verma"
    assert parsed["invoice_id"] is None
    assert parsed["has_explicit_invoice"] is False
    assert parsed["total_amount"] == 30000.0

    from src.storage.pdf_parser import generate_triples_and_chunks
    triples, chunks = generate_triples_and_chunks(parsed, 1)
    assert any(t.get("subject") == "Fourth Year Student" and t.get("predicate") == "PAID_FEE" for t in triples)
    assert any("Challan" in t.get("subject") and t.get("predicate") == "HAS_AMOUNT" for t in triples)

