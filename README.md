# RAG-Pipeline

Build a RAG Pipeline from scratch and integrated with Groq LLM.

## Overview

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline with:
- **Document Loading**: Load PDFs and text files
- **Chunking**: Split documents into semantic chunks
- **Embeddings**: Generate embeddings using sentence-transformers
- **Vector Store**: Store and retrieve documents using ChromaDB

## Installation

```bash
pip install langchain langchain-core langchain-community pypdf sentence-transformers chromadb
```

## Quick Start

```python
# Load documents
from langchain_community.document_loaders.pdf import PyPDFLoader
loader = PyPDFLoader("sample-local-pdf.pdf")
documents = loader.load()

# Split into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Create embeddings and store
embedding_manager = EmbeddingManager()
vector_store = VectorStoreManager()
embeddings = embedding_manager.generate_embeddings([doc.page_content for doc in chunks])
vector_store.add_documents(chunks, embeddings)

# Retrieve
retriever = RAGRetriever(embedding_manager, vector_store)
results = retriever.retrieve("Your query here", top_k=5)
```

## Files

- `RAG_pipeline.ipynb` - Complete implementation notebook
- `python.txt` - Sample text data
- `research1.pdf`, `research2.pdf` - Sample PDF documents

## Key Classes

- **EmbeddingManager**: Handles embedding generation with sentence-transformers
- **VectorStoreManager**: Manages ChromaDB vector storage
- **RAGRetriever**: Performs semantic search and retrieval

## Pipeline Stats

- 2 PDF documents loaded
- 563 document chunks created
- 384-dimensional embeddings
- Semantic search with cosine similarity

---

Integration with Groq LLM coming soon!
