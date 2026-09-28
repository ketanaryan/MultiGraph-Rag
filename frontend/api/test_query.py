import asyncio
import os
from src.workflow.state_machine import execute_graphrag_pipeline
from src.storage.pdf_parser import ingest_pdf_documents

class MockFile:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
    def read(self):
        with open(self.path, "rb") as f:
            return f.read()

async def main():
    print("Ingesting PDFs...")
    file_paths = [
        "sample_pdfs/Doc_1_Architecture.pdf", 
        "sample_pdfs/Doc_2_Hardware.pdf", 
        "sample_pdfs/Doc_3_Supplier.pdf"
    ]
    files = [MockFile(p) for p in file_paths]
        
    ingest_res = ingest_pdf_documents(files)
    
    print("Executing RAG Pipeline...")
    res = await execute_graphrag_pipeline(
        query="Who is the primary manufacturer responsible for fabricating the hardware unit that powers Project Aether?",
        numerical_extractions=ingest_res["numerical_extractions"],
        uploaded_documents=[f.name for f in files]
    )
    print("\nEXPECTED GROUNDED ANSWER:\n")
    print(res.get("generated_response"))

if __name__ == "__main__":
    asyncio.run(main())
