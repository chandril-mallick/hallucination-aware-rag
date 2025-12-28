import shutil
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# Import our modules
from ingest.document_loader import DocumentLoader
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from generation.generator import Generator
from evaluation.hallucination import HallucinationDetector
from evaluation.metrics import MetricsCalculator

app = FastAPI(title="Hallucination-Aware RAG API")

# Global instances (singleton pattern for simplicity)
embedder = None
vector_store = None
retriever = None
generator = None
detector = None
metrics_calc = None

DATA_DIR = "data"
INDEX_DIR = "data/faiss_index"
os.makedirs(DATA_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    text: str

class QueryResponse(BaseModel):
    answer: str
    context: List[Dict]
    hallucination_analysis: Dict
    metrics: Dict

@app.on_event("startup")
async def startup_event():
    global embedder, vector_store, retriever, generator, detector, metrics_calc
    
    # Initialize components
    embedder = Embedder()
    vector_store = VectorStore(embedder)
    
    # Try loading existing index
    if os.path.exists(INDEX_DIR):
        try:
            vector_store.load_index(INDEX_DIR)
            retriever = Retriever(vector_store)
        except Exception as e:
            print(f"Could not load index: {e}")
            
    generator = Generator(model_name="mistral") # Ensure 'mistral' is pulled in Ollama
    detector = HallucinationDetector(embedder, generator)
    metrics_calc = MetricsCalculator(embedder)
    print("System components initialized.")

@app.post("/ingest")
async def ingest_documents(files: List[UploadFile] = File(...)):
    global retriever
    
    saved_paths = []
    for file in files:
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(file_path)
    
    # Process files
    loader = DocumentLoader()
    docs = loader.load_documents(saved_paths)
    
    if not docs:
        raise HTTPException(status_code=400, detail="No documents could be loaded.")
    
    # Update Vector Store
    vector_store.create_index(docs)
    vector_store.save_index(INDEX_DIR)
    
    # Re-initialize retriever
    retriever = Retriever(vector_store)
    
    return {"message": f"Ingested {len(docs)} chunks from {len(files)} files."}

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if not retriever:
        raise HTTPException(status_code=400, detail="No documents indexed. Please ingest files first.")
    
    query = request.text
    
    # 1. Retrieve
    retrieved_docs = retriever.retrieve(query, top_k=3)
    
    # 2. Generate
    gen_result = generator.generate_answer(query, retrieved_docs)
    answer = gen_result["answer"]
    
    # 3. Detect Hallucinations
    # Run all 3 checks
    attribution = detector.check_source_attribution(answer, retrieved_docs)
    similarity = detector.check_semantic_similarity(answer, retrieved_docs)
    # verification = detector.verify_claims_agent(answer, retrieved_docs) # Optional: slow
    
    hallucination_analysis = {
        "attribution_check": attribution,
        "semantic_check": similarity,
        # "claim_verification": verification
    }
    
    # 4. Calculate API Metrics
    metrics = metrics_calc.calculate_metrics(query, answer, retrieved_docs, hallucination_analysis)
    
    # Format context for response
    context_response = []
    for doc in retrieved_docs:
        context_response.append({
            "content": doc.page_content,
            "source": os.path.basename(doc.metadata.get('source', 'unknown')),
            "page": doc.metadata.get('page', 'N/A')
        })
    
    return {
        "answer": answer,
        "context": context_response,
        "hallucination_analysis": hallucination_analysis,
        "metrics": metrics
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}
