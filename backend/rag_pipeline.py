import os
import json
from typing import Dict, List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.llms import Ollama
import sqlite3
import unicodedata
import re

# Constants for file paths and configurations
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_DIR = os.path.abspath(DATA_DIR)
DB_PATH = "metadata.db"
CHROMA_DIR = "chroma_store" 


def initialize_database():
    """Initialize SQLite database to store document metadata."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            category TEXT NOT NULL,
            title TEXT,
            date TEXT
        )
    ''')

    conn.commit()
    conn.close()


def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """Extract metadata from filename based on naming convention."""
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')

    date = None
    for part in parts:
        if '-' in part and len(part) == 10:
            date = part
            break

    return {"title": base_name, "date": date}

def clean_text(text: str) -> str:
    """Normalize text to avoid weird fonts/encodings from PDFs."""
    # Normalize to NFKC form (standard width/compatibility)
    text = unicodedata.normalize("NFKC", text)
    # Remove non-printable/control characters
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    # Collapse multiple spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_documents():
    """Load and clean PDF documents from data directory."""
    initialize_database()

    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()

    # Clean text content
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

        filename = os.path.basename(doc.metadata.get("source", "unknown"))
        source_path = doc.metadata.get("source", "")

        category = os.path.relpath(os.path.dirname(source_path), DATA_DIR)
        if category == ".":
            category = "uncategorized"

        metadata = extract_metadata_from_filename(filename)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO documents (filename, category, title, date)
            VALUES (?, ?, ?, ?)
        ''', (
            filename,
            category,
            metadata["title"],
            metadata["date"]
        ))
        conn.commit()
        conn.close()

        doc.metadata["category"] = category
        doc.metadata["title"] = metadata["title"]

    return documents


def create_chunks(documents):
    """Split loaded documents into smaller chunks for retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"
    return chunks


def create_vector_store(chunks):
    """Create or load a Chroma vector store from document chunks."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    if os.path.exists(CHROMA_DIR):
        # Load existing Chroma DB
        vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        # Create new Chroma DB
        vector_store = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=CHROMA_DIR
        )
        vector_store.persist()

    return vector_store


def setup_prompt_template():
    """Set up a context-restricted prompt template with structured output."""
    template = """
You are a financial document analysis assistant.

Use the following pieces of context to answer the question strictly.

Rules:
- Always return BOTH:
  1. A concise natural language answer
  2. A valid JSON array under the key "structured_data"
- Do NOT include comments (`//`) or explanations inside JSON
- If a field is missing, use null
- Fields in JSON must be exactly:
  ["customer_id", "customer_name", "invoice_date", "address", "postal_code", "total_amount"]

Context:
{context}

Question: {question}

Answer:
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])




def initialize_rag_pipeline():
    """Initialize and return the full RAG pipeline with Ollama as the LLM."""
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    print("Creating chunks...")
    chunks = create_chunks(documents)
    print(f"Created {len(chunks)} chunks")

    print("Creating Chroma vector store...")
    vector_store = create_vector_store(chunks)
    print("Vector store created successfully")

    prompt_template = setup_prompt_template()

    llm = Ollama(model="phi3:mini")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    print("RAG pipeline initialized successfully")

    return {"vector_store": vector_store, "qa_chain": qa_chain}


def extract_answer_and_json(raw_text: str):
    """Split answer and strictly extract JSON."""
    answer = ""
    structured_data = []

    # Split Answer vs Structured
    if "Structured Data" in raw_text:
        parts = raw_text.split("Structured Data", 1)
        answer = parts[0].replace("Answer:", "").strip()

        json_part = parts[1]
        start = json_part.find("[")
        end = json_part.rfind("]") + 1
        if start != -1 and end != -1:
            json_str = json_part[start:end]
            try:
                structured_data = json.loads(json_str)
            except Exception as e:
                print(f"Failed to parse structured data JSON: {e}")
    else:
        answer = raw_text.strip()

    return answer, structured_data



def query_documents(pipeline, question: str):
    """Run a question against the RAG pipeline and extract structured JSON if present."""
    result = pipeline["qa_chain"]({"query": question})

    answer = result.get("result", "No answer generated")
    sources = []
    structured_data = []

    # Try to extract JSON block from LLM output
    try:
        if "structured_data" in answer:
            # If model included JSON under key
            parsed = json.loads(answer.split("structured_data")[-1])
            structured_data = parsed
        else:
            # Try to find JSON manually in the answer
            start = answer.find("[")
            end = answer.rfind("]") + 1
            if start != -1 and end != -1:
                structured_data = json.loads(answer[start:end])
    except Exception as e:
        print(f"Could not parse structured JSON: {e}")

    if "source_documents" in result:
        for doc in result["source_documents"]:
            short_filename = os.path.basename(doc.metadata.get("source", "unknown"))
            sources.append({
                "filename": short_filename,
                "category": doc.metadata.get("category", "unknown"),
                "title": doc.metadata.get("title", "Unknown Title"),
            })

    return {"answer": answer, "sources": sources, "structured_data": structured_data}

