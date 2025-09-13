from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import traceback

# Import our custom modules
from backend.rag_pipeline import initialize_rag_pipeline, query_documents

# Global RAG pipeline instance
rag_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI startup/shutdown."""
    global rag_pipeline
    try:
        print("🔄 Initializing RAG pipeline...")
        rag_pipeline = initialize_rag_pipeline()
        print("✅ RAG pipeline initialized successfully.")
        yield
    except Exception as e:
        print(f"❌ Failed to initialize RAG pipeline: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during initialization")
    finally:
        print("🛑 Application shutdown...")


# Create app with lifespan
app = FastAPI(
    title="Enterprise Document Q&A Assistant",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------
# Models
# ---------------------------

class QueryRequest(BaseModel):
    """Incoming query payload."""
    question: str


class StructuredItem(BaseModel):
    """Optional structured record parsed from documents."""
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    invoice_date: Optional[str] = None
    address: Optional[str] = None
    postal_code: Optional[str] = None
    total_amount: Optional[float] = None


class QueryResponse(BaseModel):
    """Response payload returned to frontend."""
    answer: str
    sources: list
    structured_data: Optional[List[StructuredItem]] = None


# ---------------------------
# Endpoints
# ---------------------------

@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Enterprise Document Q&A Assistant is running 🚀"}


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    Handle user queries by retrieving relevant documents and generating answers.

    Returns both a natural-language answer and (if available) structured data
    like invoice/order rows that Streamlit can render as a table.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")

    try:
        result = query_documents(rag_pipeline, request.question)

        # Base response
        response = {
            "answer": result["answer"],
            "sources": result["sources"],
            "structured_data": None
        }

        # Example: if query_documents attaches structured JSON, pass it through
        if "structured_data" in result and result["structured_data"]:
            response["structured_data"] = [
                StructuredItem(**item) for item in result["structured_data"]
            ]

        return response
    except Exception as e:
        error_text = f"Error processing query: {e}\n{traceback.format_exc()}"
        print("❌", error_text)
        raise HTTPException(status_code=500, detail=error_text)


# ---------------------------
# Run standalone (optional)
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

