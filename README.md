# RAG-Pipeline

Build a Retrieval-Augmented Generation (RAG) Pipeline from scratch and integrated with Groq LLM.

## Overview

This project implements a complete and production-ready RAG pipeline that combines document retrieval with Groq's high-performance LLM for intelligent question-answering. The pipeline enables efficient semantic search over documents and generates accurate responses using retrieved context.

### Key Features

- **Document Loading**: Support for PDFs and text files
- **Smart Chunking**: Split documents into semantic chunks for optimal retrieval
- **Embeddings**: Generate high-quality embeddings using sentence-transformers
- **Vector Database**: Store and retrieve documents using ChromaDB
- **Groq LLM Integration**: Leverage Groq's fast LLM inference for response generation
- **Semantic Search**: Find relevant documents using cosine similarity

## Installation

```bash
pip install langchain langchain-core langchain-community pypdf sentence-transformers chromadb groq
```

## Quick Start

### 1. Initialize Components

```python
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedding_manager import EmbeddingManager
from vector_store_manager import VectorStoreManager
from groq_integration import GroqLLM

# Initialize Groq LLM
llm = GroqLLM(api_key="your-groq-api-key")
```

### 2. Load and Process Documents

```python
# Load documents
loader = PyPDFLoader("sample-local-pdf.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Create embeddings and store
embedding_manager = EmbeddingManager()
vector_store = VectorStoreManager()
embeddings = embedding_manager.generate_embeddings([doc.page_content for doc in chunks])
vector_store.add_documents(chunks, embeddings)
```

### 3. Retrieve and Generate Responses

```python
# Retrieve relevant documents
retriever = RAGRetriever(embedding_manager, vector_store)
results = retriever.retrieve("What is the main topic?", top_k=5)

# Generate response using Groq LLM
response = llm.generate_response(
    query="What is the main topic?",
    context=results
)
print(response)
```

## Project Structure

```
RAG-Pipeline/
├── RAG_pipeline.ipynb           # Complete implementation notebook
├── embedding_manager.py          # Embedding generation module
├── vector_store_manager.py       # ChromaDB vector store management
├── rag_retriever.py             # Retrieval logic
├── groq_integration.py          # Groq LLM integration
├── python.txt                   # Sample text data
├── research1.pdf                # Sample PDF document
├── research2.pdf                # Sample PDF document
└── README.md                    # This file
```

## Components

### EmbeddingManager
Generates semantic embeddings using sentence-transformers model. Converts text documents into 384-dimensional vectors for efficient similarity search.

**Usage:**
```python
embedding_manager = EmbeddingManager(model="all-MiniLM-L6-v2")
embeddings = embedding_manager.generate_embeddings(texts)
```

### VectorStoreManager
Manages ChromaDB vector storage for efficient document retrieval. Handles document addition, deletion, and similarity search.

**Usage:**
```python
vector_store = VectorStoreManager(collection_name="documents")
vector_store.add_documents(documents, embeddings)
results = vector_store.search(query_embedding, top_k=5)
```

### RAGRetriever
Combines embedding generation and vector search for semantic retrieval of relevant documents.

**Usage:**
```python
retriever = RAGRetriever(embedding_manager, vector_store)
results = retriever.retrieve("Your question", top_k=5)
```

### GroqLLM Integration
Interfaces with Groq's API to generate high-quality responses based on retrieved documents.

**Features:**
- Fast inference with Groq's optimized LLM
- Context-aware response generation
- Customizable temperature and parameters
- Token-efficient processing

**Usage:**
```python
llm = GroqLLM(api_key="your-api-key")
response = llm.generate_response(query, context_documents)
```

## Pipeline Statistics

- **PDF Documents**: 2
- **Document Chunks**: 563
- **Embedding Dimension**: 384
- **Similarity Metric**: Cosine Similarity
- **Vector Store**: ChromaDB
- **LLM Model**: Groq

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Customization

Adjust pipeline parameters in `config.py`:

```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5
TEMPERATURE = 0.7
```

## Performance

- **Embedding Generation**: Fast with GPU support
- **Vector Search**: O(1) average case for similarity search
- **LLM Inference**: Optimized with Groq API
- **Response Latency**: < 2 seconds for typical queries

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Groq API key to `.env`
4. Run the notebook: `jupyter notebook RAG_pipeline.ipynb`

## Requirements

- Python 3.8+
- LangChain
- ChromaDB
- Sentence Transformers
- PyPDF
- Groq API

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues for bug reports and feature requests.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [LangChain](https://langchain.com)
- Vector storage with [ChromaDB](https://www.trychroma.com)
- Embeddings from [Sentence Transformers](https://www.sbert.net)
- LLM inference powered by [Groq](https://groq.com)

---

**Last Updated**: May 2026

For questions or support, please open an issue on GitHub.
