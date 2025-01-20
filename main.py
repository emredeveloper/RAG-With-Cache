# main.py
from config import Config
from embedding_model import EmbeddingModel
from retriever import Retriever
from language_model import LanguageModel
from rag_system import RAGSystem
import pandas as pd
import inquirer  # İnteraktif seçim kutusu için

def select_option(options, prompt):
    """
    Kullanıcıdan bir seçenek seçmesini ister.
    """
    questions = [
        inquirer.List('choice', message=prompt, choices=options)
    ]
    answers = inquirer.prompt(questions)
    return answers['choice']

def main():
    # Yapılandırma ayarlarını yükle
    config = Config()

    # Kullanıcıdan embedding modeli seçmesini iste
    embedding_model_choice = select_option(
        config.EMBEDDING_MODELS,
        "Lütfen bir embedding modeli seçin:"
    )
    embedding_model_name = embedding_model_choice

    # Kullanıcıdan dil modeli seçmesini iste
    language_model_choice = select_option(
        config.LANGUAGE_MODELS,
        "Lütfen bir dil modeli seçin:"
    )
    language_model_name = language_model_choice

    # Embedding modelini yükle
    embedding_model = EmbeddingModel(embedding_model_name)

    # Retriever'ı yükle ve indeksi oluştur
    retriever = Retriever(embedding_model)

    # truthful_qa veri setini yükle
    truthful_qa_df = pd.read_pickle('truthful_qa_validation.pkl')

    # Belgeleri hazırla (soruları belge olarak kullan)
    documents = truthful_qa_df['question'].tolist()
    retriever.build_index(documents)

    # Dil modelini yükle
    language_model = LanguageModel(language_model_name)

    # RAG sistemini oluştur
    rag_system = RAGSystem(embedding_model, retriever, language_model)

    # Örnek sorgu
    query = "What is the capital of France?"
    answer = rag_system.answer_question(query, top_k=config.TOP_K)
    print("RAG Answer:", answer)

if __name__ == "__main__":
    main()