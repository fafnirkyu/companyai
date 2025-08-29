# 🏢 Enterprise Document Q&A Assistant

**Enterprise Document Q&A Assistant** is an AI-powered system that lets you **query company documents** (Invoices, Purchase Orders, Shipping Orders, etc.) in natural language.  
It uses **Retrieval-Augmented Generation (RAG)** with **FastAPI**, **Streamlit**, **LangChain**, **ChromaDB**, and **Ollama** to provide accurate, explainable answers backed by document sources.  

---

## ✨ Features
- 🔍 Ask questions about your company PDFs in plain English  
- 📂 Automatic ingestion of documents from the `data/` folder  
- 🧠 RAG pipeline with **embeddings + local LLM** for grounded answers  
- 📊 Structured JSON outputs (when applicable)  
- ⬇ Export results to **Excel/CSV** directly from the UI  
- 🚀 One-command startup (`python run_app.py`)  

---

## 🏗️ Project Structure
Companyai/
│
├── backend/
│ ├── main.py # FastAPI app (API endpoints)
│ ├── rag_pipeline.py # RAG pipeline (loading, embeddings, querying)
│
├── frontend/
│ ├── streamlit_app.py # Streamlit frontend (chat UI + downloads)
│
├── data/ # Place your PDF files here
│
├── run_app.py # Orchestrator: starts backend + frontend
├── requirements.txt # Python dependencies
└── README.md

---

## ⚙️ How It Works

### 1. **Document Processing**
- PDFs inside `data/` are loaded with LangChain’s `DirectoryLoader`.
- Split into **chunks** (~1000 characters each).
- Each chunk is embedded using **HuggingFace sentence-transformers**.
- Embeddings are stored in a **Chroma vector store** for semantic retrieval.
- Metadata (filename, category, date) is logged in **SQLite**.

### 2. **Backend (FastAPI)**
- Provides two endpoints:
  - `GET /` → Health check  
  - `POST /query` → Takes a natural language question and returns:
    - 📝 Answer in plain English  
    - 📄 Sources used for retrieval  
    - 📊 Structured JSON (when available)  

The backend uses **LangChain’s RetrievalQA** with **Ollama** as the LLM.

### 3. **Frontend (Streamlit)**
- A chat-like interface for interacting with documents.
- Shows:
  - Assistant’s answer  
  - Sources  
  - Structured data as an interactive table  
- Download results directly as **Excel/CSV**.

### 4. **Unified Startup**
- Run both backend + frontend with a single command:

```bash
python run_app.py
```
The script:

Starts FastAPI with Uvicorn

Waits until it’s ready

Launches Streamlit

🚀 Quickstart
1. Clone the repo
```bash
git clone https://github.com/your-username/companyai.git
cd companyai
```
2. Create virtual environment & install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```
```bash
pip install -r requirements.txt
```
3. Add your documents
Put your company PDFs inside the data/ folder.

4. Run the assistant
```bash
python run_app.py
```
5. Ask questions
Example queries:

“Show me our latest invoices”

“Which customers spent the most last quarter?”

“List all orders shipped to Canada.”

📊 Example Output
User:

perl
Show me our latest invoices
Assistant:
The most recent invoices were from customer GREAL on April 7, 2018 and January 6, 2018.

Structured JSON:

json
[
  {
    "customer_id": "GREAL",
    "invoice_date": "2018-04-07",
    "address": "2732 Baker Blvd.",
    "postal_code": "97403",
    "total_amount": 391.58
  },
  {
    "customer_id": "GREAL",
    "invoice_date": "2018-01-06",
    "address": "2732 Baker Blvd.",
    "postal_code": "97403",
    "total_amount": 8891.0
  }
]
Downloadable as Excel or CSV.

🛠️ Tech Stack
Frontend: Streamlit

Backend: FastAPI + Uvicorn

RAG/NLP: LangChain + HuggingFace embeddings

LLM: Ollama (local models like phi3:mini)

Vector DB: Chroma

Database: SQLite

📌 Future Improvements
🔑 Authentication / multi-user support

📑 Support for more file formats (Word, Excel, CSV)

📊 Dashboard view with charts & KPIs


☁ Deployment with Docker + Kubernetes
