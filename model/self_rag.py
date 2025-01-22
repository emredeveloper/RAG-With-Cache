from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import json
from dotenv import load_dotenv
import os
import warnings

# Uyarıları gizle
warnings.filterwarnings("ignore")

# .env dosyasından API anahtarını yükle
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

class Self_RAG:
    def __init__(self, pdf_files):
        # Model yapılandırması
        self.generation_config = {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json"
        }
        
        # JSON çıktısı için model
        self.llm_json = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=self.generation_config,
        )
        
        # Normal metin çıktısı için model
        self.llm = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
        # Embedding modeli (API anahtarı manuel olarak geçiliyor)
        self.embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key  # API anahtarı burada manuel olarak geçiliyor
        )
        
        # Vektör deposu ve diğer ayarlar
        self.vectorstore = None
        self.pdf_files = pdf_files  # PDF dosyalarının listesi
        self.chunk_size = 500  # Metin parçalama boyutu

    def load_and_split(self):
        """Tüm PDF dosyalarını yükler ve metni parçalara ayırır."""
        all_docs = []
        for pdf_file in self.pdf_files:
            # PDF dosyasının tam yolunu oluştur
            full_path = os.path.join("data", "pdfs", pdf_file)
            
            # PDF dosyasını yükle
            loader = PyPDFLoader(full_path)
            data = loader.load()
            
            # Metni parçalara ayır
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size)
            docs = text_splitter.split_documents(data)
            
            # Her bir dokümana kaynak dosya adını ekle
            for doc in docs:
                doc.metadata["source_file"] = pdf_file
            
            all_docs.extend(docs)
        return all_docs

    def vector_store(self, docs):
        """Belgeleri vektör deposuna kaydeder."""
        self.vectorstore = FAISS.from_documents(
            documents=docs,
            embedding=self.embedding
        )
        self.vectorstore.save_local("multi_pdf_index")  # Vektör deposunu yerel olarak kaydet
        return self.vectorstore

    def retrieve_doc(self, query, k=3):
        """Sorguya uygun belgeleri alır."""
        if not self.vectorstore:
            self.vectorstore = FAISS.load_local(
                "multi_pdf_index",
                embeddings=self.embedding,
                allow_dangerous_deserialization=True
            )
        
        # En alakalı belgeleri al
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        relevant_docs = retriever.invoke(query)
        
        # Belge içeriklerini ve kaynak dosyalarını birleştir
        relevant_content = "\n".join(doc.page_content for doc in relevant_docs)
        source_files = ", ".join(set(doc.metadata["source_file"] for doc in relevant_docs))
        
        return relevant_content, source_files

    def evaluate_response_quality(self, query, response, context):
        """Yanıtın kalitesini değerlendirir."""
        eval_prompt = f"""
        Evaluate the quality of this response to the given query.
        Query: {query}
        Context: {context}
        Response: {response}

        Score the following aspects from 1-10:
        1. Relevance to query
        2. Factual accuracy based on context
        3. Completeness of answer
        4. Coherence and clarity

        **Return the evaluation **ONLY as a JSON object.** Do not include any other text or explanation. Strictly adhere to this format:
        *format:*
        {{
            "evaluation": {{
                "relevance": {{
                    "score": 9,
                    "explanation": "Explanation here."
                }},
                "factual_accuracy": {{
                    "score": 8,
                    "explanation": "Explanation here."
                }},
                "completeness": {{
                    "score": 7,
                    "explanation": "Explanation here."
                }},
                "coherence": {{
                    "score": 10,
                    "explanation": "Explanation here."
                }},
                "overall_score": 8.5,
                "overall_explanation":"Explanation here"
            }}
        }}
        """
        try:
            # JSON çıktısı oluştur
            response = self.llm_json.generate_content(eval_prompt)
            
            # JSON'u ayrıştır
            evaluation_data = json.loads(response.text)
            
            # Genel puan ve açıklamayı al
            overall_score = evaluation_data["evaluation"]["overall_score"]
            overall_explanation = evaluation_data["evaluation"]["overall_explanation"]
            
            return overall_score, overall_explanation
        except json.JSONDecodeError as e:
            print("JSONDecodeError:", e)
            print("Response Text for Debugging:", response.text)
            return None
        except Exception as e:
            print("Error occurred:", e)
            return None

    def improve_response(self, query, initial_response, evaluation, feedback):
        """Yanıtı değerlendirme geri bildirimine göre iyileştirir."""
        improve_prompt = f"""
        Improve this response based on the evaluation feedback:
        Query: {query}
        Initial Response: {initial_response}
        Evaluation: {evaluation}
        Feedback: {feedback}
        
        Generate an improved response that addresses the weaknesses identified.
        """
        
        # İyileştirilmiş yanıt oluştur
        improved_response = self.llm.generate_content(improve_prompt)
        return improved_response.text

    def generate_response(self, query):
        """Self-RAG yaklaşımıyla yanıt oluşturur."""
        # İlgili belgeleri al
        context, source_files = self.retrieve_doc(query)
        
        # İlk yanıtı oluştur
        initial_prompt = f"""
        Based on the following context, answer the query:
        Query: {query}
        Context: {context}
        
        Provide a comprehensive and accurate response.
        """
        initial_response = self.llm.generate_content(initial_prompt).text
        
        # Yanıtın kalitesini değerlendir
        evaluation, explanation = self.evaluate_response_quality(query, initial_response, context)
        
        # Eğer puan 8'den düşükse yanıtı iyileştir
        if evaluation < 8:
            improved_response = self.improve_response(query, initial_response, evaluation, explanation)
            return {
                "final_response": improved_response,
                "evaluation": evaluation,
                "improved": True,
                "source_files": source_files
            }
        
        # Puan yeterliyse ilk yanıtı döndür
        return {
            "final_response": initial_response,
            "evaluation": evaluation,
            "improved": False,
            "source_files": source_files
        }

if __name__ == "__main__":
    # PDF dosyalarının listesi
    pdf_files = [
        "2405.12981v1.pdf",
        "2411.19865v1.pdf",
        "2412.08905v1.pdf",
        "2412.13663v2.pdf",
        "2501.00663v1.pdf",
        "2501.04040v1.pdf",
        "2501.06252v2.pdf"
    ]
    
    # Self_RAG örneği oluştur
    rag = Self_RAG(pdf_files)
    
    # PDF'leri yükle ve parçalara ayır
    docs = rag.load_and_split()
    
    # Vektör deposunu oluştur
    vectorstore = rag.vector_store(docs)
    
    # Sorgu oluştur ve yanıt al
    query = "what is self adaptive llm"
    response = rag.generate_response(query)
    
    # Yanıtı yazdır
    print(json.dumps(response, indent=4))