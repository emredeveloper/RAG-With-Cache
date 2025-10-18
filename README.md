# 🚀 RAG-With-Cache

A compact reference implementation of **Retrieval-Augmented Generation (RAG)**
that focuses on the core building blocks: PDF ingestion, vector search, and
language model generation. The codebase has been simplified so the main RAG
pipeline lives in a dedicated `rag/` package with minimal dependencies and a
clean entry point.

---

## 🌟 Features

- **Modular package layout** – Embedding, language, data-loading, and
  retriever utilities are organised under `rag/` for easier reuse.
- **FAISS and HyDE retrievers** – Switch between traditional dense retrieval
  and HyDE-style hypothetical document retrieval from the command line.
- **Model caching** – Embedding and language models are cached locally to
  avoid repeated downloads.
- **PDF utilities** – Lightweight helpers for loading and chunking documents.
- **Test coverage** – Pytest suite that exercises each major component with
  fast stubs.

---

## 🚀 Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your documents**

   Place PDFs inside `data/pdfs/`. The directory is created automatically when
   running the CLI, but adding files ahead of time lets you test retrieval
   immediately.

3. **Run the demo CLI**

   ```bash
   python main.py
   ```

   Choose between FAISS or HyDE retrieval when prompted and start asking
   questions about your documents.

---

## 📁 Project Structure

```text
├── rag/
│   ├── config.py             # Configuration dataclass
│   ├── data/                 # PDF loading utilities
│   ├── embeddings.py         # SentenceTransformer wrapper with caching
│   ├── language.py           # HuggingFace causal LM wrapper with caching
│   ├── retrievers/           # FAISS and HyDE retrievers
│   └── system.py             # High-level RAG orchestration
├── data/pdfs/                # PDF documents for retrieval
├── tests/                    # Pytest suite
├── main.py                   # Command-line entry point
├── requirements.txt          # Minimal dependency set
└── setup.py                  # Package metadata
```

---

## 🧪 Running the Tests

```bash
pytest
```

The HyDE tests rely on stubbed models so they run quickly without downloading
large checkpoints.
