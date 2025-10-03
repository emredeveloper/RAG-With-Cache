# 🚀 RAG-Best-Practices

A comprehensive study and implementation of best practices for **Retrieval-Augmented Generation (RAG)** systems. This project includes advanced features like **model caching**, **multilingual support**, **evaluation metrics**, and **Agentic RAG** with web search integration.

---

## 🌟 Features

### 🤖 Agentic RAG System (`model/agentic_rag_ollama.py`)

- **Intelligent Query Routing**: Automatically determines whether to search local PDFs, web, or both
- **Selective Search Modes**: Use "sadece PDF" for PDF-only search or "sadece web" for web-only search
- **Web Search Integration**: DuckDuckGo-powered web content retrieval with source attribution
- **Advanced Chunking**: Chonkie-powered text chunking (Token, Table, Semantic chunkers)
- **Persistent Vector Storage**: ChromaDB for reliable document storage and retrieval
- **Rich Terminal UI**: Beautiful console interface with panels and markdown rendering
- **Table & Figure Extraction**: Automatic extraction of tables and figures from PDFs
- **Dynamic Prompting**: Context-aware prompts based on query type (summary, analysis, comparison, etc.)
- **HTML Export**: Export results to styled HTML files for sharing and archiving
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

### 📚 Core RAG Components

- **Advanced Retrieval Options**:
  - **FAISS**: High-performance vector search for fast and accurate document retrieval.
  - **LangChain**: Intelligent search with seamless integration of language models.
  - **HyDE (Hypothetical Document Embeddings)**: Generate hypothetical documents to improve retrieval relevance.

- **Model Caching**: Efficient caching of embedding and language models to reduce load times and improve performance.

- **Multilingual Support**: Built-in support for multilingual embeddings, enabling cross-lingual document retrieval.

- **Evaluation Metrics**: Comprehensive evaluation tools to measure retrieval accuracy and generation quality.

- **Customizable Configuration**: Easily adjust chunk sizes, overlap, and retrieval parameters to suit your needs.

- **Table Augmented Generation (TAG)**: Involves generating textual outputs based on structured table data. TAG systems analyze tables (e.g., CSV, Excel) and use this structured information to produce contextually relevant text outputs. These systems are useful in scenarios such as data-to-text generation for business intelligence reports or dynamic documentation based on tabular data.

---

### 📊 Tagging and Information

- **#RAG**: Related to the concept of Retrieval-Augmented Generation.
- **#AgenticRAG**: Intelligent RAG with autonomous decision-making and multi-source retrieval.
- **#FAISS**: Refers to the FAISS library for fast document retrieval.
- **#LangChain**: A tool for intelligent search integration with language models.
- **#HyDE**: A method to generate hypothetical document embeddings to improve retrieval relevance.
- **#ModelCaching**: Caching techniques to improve performance by reducing load times.
- **#MultilingualSupport**: Enabling support for multiple languages in document retrieval and generation.
- **#EvaluationMetrics**: Metrics to evaluate the accuracy and performance of RAG systems.
- **#TAG**: Refers to **Table Augmented Generation**, where structured tabular data is used to generate contextually relevant text outputs.
- **#ChromaDB**: Persistent vector database for reliable document storage.
- **#Ollama**: Local LLM inference for privacy and cost-efficiency.
- **#WebSearch**: Integration with web search APIs for up-to-date information.

---

## 🚀 Quick Start

### Agentic RAG System

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Ollama is Running**:

   ```bash
   # Pull required models
   ollama pull granite4:tiny-h
   ollama pull embeddinggemma:latest
   ```

3. **Run the Agentic RAG System**:

   ```bash
   python model/agentic_rag_ollama.py
   ```

4. **Query Examples**:
   - Regular query: "What is artificial intelligence?"
   - PDF-only search: "sadece PDF: Explain machine learning algorithms"
   - Web-only search: "sadece web: Latest developments in AI"
   - Export results: After getting an answer, type 'y' when prompted to export to HTML

### Features Overview

- **Intelligent Routing**: The system automatically decides whether to search your local PDFs, the web, or both
- **Selective Search**: Use "sadece PDF" or "sadece web" prefixes for targeted searches
- **Rich Output**: Beautiful terminal interface with markdown rendering
- **HTML Export**: Save results as styled HTML files for sharing
- **Source Attribution**: All answers include source information ([PDF: filename] or [WEB: url])
- **Table Extraction**: Automatically extracts and displays tables from PDFs
- **Persistent Storage**: Your document database is saved between sessions

---

## 📁 Project Structure

```text
├── model/
│   ├── agentic_rag_ollama.py    # Main Agentic RAG implementation
│   ├── embedding_model.py       # Embedding model utilities
│   └── ...
├── data/
│   └── pdfs/                    # PDF documents for RAG
├── examples/
│   └── streamlit/               # Streamlit web interface examples
├── tests/                       # Unit tests
└── requirements.txt             # Python dependencies
```
