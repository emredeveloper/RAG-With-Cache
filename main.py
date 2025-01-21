from config import Config
from model.embedding_model import EmbeddingModel
from model.retriever_factory import RetrieverFactory
from model.language_model import LanguageModel
from model.rag_system import RAGSystem
from data_loader.pdf_loader import PDFLoader
import inquirer

def select_option(options, prompt):
    questions = [inquirer.List('choice', message=prompt, choices=options)]
    answers = inquirer.prompt(questions)
    return answers['choice']

def main():
    config = Config()
    
    # PDF'leri yükle ve parçala
    pdf_loader = PDFLoader(config.PDF_DIRECTORY)
    raw_texts = pdf_loader.load_pdfs()
    documents = pdf_loader.chunk_text(raw_texts, config.CHUNK_SIZE)
    
    # Retriever seçimi (LangChain kaldırıldı, sadece FAISS kullanılacak)
    retriever_choice = select_option(
        config.RETRIEVER_OPTIONS, 
        "Retriever tipini seçin:"
    )
    config.DEFAULT_RETRIEVER = retriever_choice

    # Sistem bileşenlerini yükle
    embedding_model = EmbeddingModel(config.DEFAULT_EMBEDDING_MODEL)
    retriever = RetrieverFactory.create_retriever(
        config, 
        config.DEFAULT_EMBEDDING_MODEL, 
        documents
    )
    
    language_model = LanguageModel(config.DEFAULT_LANGUAGE_MODEL)
    rag_system = RAGSystem(embedding_model, retriever, language_model)

    # Etkileşimli sorgu döngüsü
    while True:
        query = input("\nSoru girin (çıkmak için 'q'): ")
        if query.lower() == 'q':
            break
        answer = rag_system.answer_question(query, top_k=config.TOP_K)
        print("\nCevap:", answer)

if __name__ == "__main__":
    main()