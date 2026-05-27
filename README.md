# 🔍 RAG Pipeline — Retrieval-Augmented Generation

An end-to-end **Retrieval-Augmented Generation (RAG)** pipeline built from scratch using LangChain, ChromaDB, Sentence Transformers, and Groq's LLM API.

---

## What is RAG?

RAG is a technique that enhances LLM responses by first retrieving relevant information from a knowledge base, then using that context to generate accurate, grounded answers — reducing hallucinations and keeping responses factual.

---

## How It Works

The pipeline runs in two stages:

**Ingestion** — Documents are loaded, split into overlapping chunks, converted into vector embeddings, and stored persistently in ChromaDB.

**Retrieval & Generation** — A user query is embedded, matched against stored chunks via cosine similarity, and the top results are passed as context to the LLM to generate a final answer.

```
PDF Files → Chunks → Embeddings → ChromaDB
                                      ↓
  Answer  ←  Groq LLM  ←  Retrieved Context  ←  Query
```

---

## Pipeline Stages

**1. Document Loading** — Supports PDF and plain text files via LangChain's `PyPDFLoader` and `TextLoader`. All PDFs in a target folder are loaded automatically.

**2. Chunking** — Documents are split using `RecursiveCharacterTextSplitter` with a chunk size of 500 characters and 50-character overlap to preserve context at boundaries.

**3. Embedding** — Each chunk is encoded into a dense vector using `all-MiniLM-L6-v2` from Sentence Transformers — a lightweight model with strong semantic understanding.

**4. Vector Store** — Embeddings are stored in a persistent ChromaDB collection, so documents only need to be embedded once and survive across sessions.

**5. Retrieval** — The `RAGRetriever` class embeds the user query, performs a semantic search, and ranks results by cosine similarity, returning the top-k most relevant chunks.

**6. Generation** — Retrieved chunks are injected into a prompt template and sent to `qwen/qwen3-32b` via the Groq API, which generates a context-grounded answer.

---

## Tech Stack

| Component | Tool |
|---|---|
| Document Loading | LangChain (`PyPDFLoader`, `TextLoader`) |
| Chunking | `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (persistent) |
| Similarity | Cosine Similarity via `scikit-learn` |
| LLM | Groq API — `qwen/qwen3-32b` |

---

## Getting Started

**Prerequisites:** Python 3.8+, a [Groq API key](https://console.groq.com/)

```bash
git clone https://github.com/your-username/rag-pipeline.git
cd rag-pipeline
pip install langchain langchain-core langchain-community pypdf pymupdf \
            sentence-transformers chromadb langchain-groq scikit-learn
```

Add your Groq API key in the notebook, place your PDFs in `sample_data/pdfs/`, and run all cells.

---
