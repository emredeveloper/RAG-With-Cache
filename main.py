"""Command-line interface for experimenting with the RAG pipeline."""

from __future__ import annotations

from typing import Sequence, Tuple

import inquirer

from rag import Config, EmbeddingModel, LanguageModel, RAGSystem
from rag.data import PDFLoader
from rag.retrievers import FaissRetriever, HyDERetriever


def _select_option(options: Sequence[Tuple[str, str]], prompt: str) -> str:
    question = inquirer.List("choice", message=prompt, choices=list(options))
    answers = inquirer.prompt([question])
    if answers is None:
        raise KeyboardInterrupt
    return answers["choice"]


def main() -> None:
    config = Config()
    config.ensure_pdf_directory()

    pdf_loader = PDFLoader(config.pdf_directory)
    raw_texts = pdf_loader.load_pdfs()
    documents = pdf_loader.chunk_text(raw_texts, config.chunk_size)

    retriever_key = _select_option(config.retriever_options, "Select the retriever type:")
    embedding_model = EmbeddingModel(config.default_embedding_model)

    if retriever_key == "faiss":
        retriever = FaissRetriever(embedding_model)
        retriever.build_index(documents)
    elif retriever_key == "hyde":
        hyde_config = config.hyde_settings
        retriever = HyDERetriever(
            files_path=str(config.pdf_directory),
            chunk_size=hyde_config["chunk_size"],
            chunk_overlap=hyde_config["chunk_overlap"],
            language_model_name=hyde_config["language_model"],
            embedding_model_name=hyde_config["embedding_model"],
        )
    else:
        raise ValueError(f"Unsupported retriever option: {retriever_key}")

    language_model = LanguageModel(config.default_language_model)
    rag_system = RAGSystem(embedding_model, retriever, language_model)

    while True:
        try:
            query = input("\nEnter a question (press 'q' to quit): ")
        except EOFError:
            break
        if query.lower() == "q":
            break
        answer = rag_system.answer_question(query, top_k=config.top_k)
        print("\nAnswer:", answer)


if __name__ == "__main__":
    main()
