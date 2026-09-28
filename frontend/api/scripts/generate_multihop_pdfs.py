import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# Target directory for generated sample PDFs
output_dir = "sample_pdfs"
os.makedirs(output_dir, exist_ok=True)

styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    "TitleStyle", parent=styles["Heading1"], fontSize=18, spaceAfter=12
)
body_style = ParagraphStyle(
    "BodyStyle", parent=styles["Normal"], fontSize=12, leading=16, spaceAfter=10
)


def create_pdf(filename, title, content):
    filepath = os.path.join(output_dir, filename)
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    story = [Paragraph(title, title_style), Spacer(1, 12)]
    for paragraph in content:
        story.append(Paragraph(paragraph, body_style))
        story.append(Spacer(1, 8))
    doc.build(story)
    print(f"[SUCCESS] Generated: {filepath}")


# Dataset 1: System Hardware Multi-Hop Chain
doc1_content = [
    "This document outlines the high-level system architecture for Project Aether.",
    "Project Aether operates on top of the central compute engine designated as Subsystem-X9.",
    "All security protocols and memory allocations for Project Aether are delegated directly to Subsystem-X9.",
]
create_pdf(
    "Doc_1_Architecture.pdf", "System Architecture Document", doc1_content
)

doc2_content = [
    "This technical manual details hardware specifications for core subsystems.",
    "Subsystem-X9 is powered exclusively by the custom-designed silicon unit known as the Helios Processing Core.",
    "The Helios Processing Core manages high-throughput matrix multiplication tasks for all attached modules.",
]
create_pdf(
    "Doc_2_Hardware.pdf", "Hardware Configuration Manual", doc2_content
)

doc3_content = [
    "This report lists vendor procurement details for specialized hardware components.",
    "The Helios Processing Core is manufactured under exclusive enterprise agreement by QuantumTech Foundry.",
    "QuantumTech Foundry handles all fabrication, wafer manufacturing, and assembly for Helios units.",
]
create_pdf(
    "Doc_3_Supplier.pdf", "Supply Chain & Procurement Report", doc3_content
)

# Dataset 2: Financial Compliance Multi-Hop Chain
docA_content = [
    "Under Master Service Agreement MSA-2026, Enterprise Corp receives cloud infrastructure services from Provider Apex.",
    "All billings under MSA-2026 are processed under Schedule B rates.",
]
create_pdf("Doc_A_Agreement.pdf", "Master Service Agreement", docA_content)

docB_content = [
    "Schedule B rate card specifies that high-compute instances incur charges tied to Energy Tier-3 billing.",
    "Energy Tier-3 rate adjustments are audited under Compliance Protocol CP-88.",
]
create_pdf("Doc_B_Schedule.pdf", "Schedule B Rate Card", docB_content)

docC_content = [
    "Compliance Protocol CP-88 requires mandatory quarterly auditing conducted by Apex Auditing Group.",
    "Apex Auditing Group issues compliance certifications upon successful review.",
]
create_pdf("Doc_C_Compliance.pdf", "Compliance Protocol CP-88", docC_content)

print(
    "\nAll 6 interlinked PDFs successfully created in 'sample_pdfs/' directory!"
)
