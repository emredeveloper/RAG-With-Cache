"""Wrapper to run the Ollama Streamlit app from examples.

This file intentionally keeps the original app code in `model/ollama_rag_local.py`.
Run with:
    streamlit run examples/streamlit/run_ollama_rag_local.py

Importing the module executes the top-level Streamlit app defined in the original file.
"""
from model import ollama_rag_local  # noqa: F401  (module has top-level Streamlit code)
