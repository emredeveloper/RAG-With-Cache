class RAGSystem:
    def __init__(self, embedding_model, retriever, language_model):
        """
        RAG sistemini başlatır.

        Args:
            embedding_model: Metinleri vektörlere dönüştürmek için kullanılan embedding modeli.
            retriever: Belge getirme işlemini gerçekleştiren retriever.
            language_model: Sorulara cevap üretmek için kullanılan dil modeli.
        """
        self.embedding_model = embedding_model
        self.retriever = retriever
        self.language_model = language_model
    
    def answer_question(self, query: str, top_k: int = 2) -> str:
        """
        Verilen bir soruya cevap üretir.

        Args:
            query: Soru metni.
            top_k: Getirilecek en benzer belge sayısı.

        Returns:
            str: Sorunun cevabı.
        """
        try:
            # Retriever'dan benzer belgeleri ve hipotetik belgeyi al
            similar_docs, hypothetical_doc = self.retriever.retrieve(query, top_k)
            
            # Benzer belgeleri kullanarak prompt oluştur
            prompt = self._create_prompt(query, similar_docs)
            
            # Dil modeli ile cevap üret
            answer = self.language_model.generate(prompt)
            return answer
        except Exception as e:
            # Hata durumunda kullanıcıya bilgi ver
            print(f"Soru cevaplanırken bir hata oluştu: {str(e)}")
            return "Üzgünüm, bu soruyu cevaplayamadım."
    
    def _create_prompt(self, query: str, similar_docs: list) -> str:
        """
        Soru ve benzer belgeleri kullanarak dil modeli için bir prompt oluşturur.

        Args:
            query: Soru metni.
            similar_docs: Benzer belgeler listesi (belge metni ve benzerlik skoru içeren tuple'lar).

        Returns:
            str: Dil modeli için hazırlanmış prompt.
        """
        try:
            # Benzer belgeleri formatla
            context = "\n".join(
                f"[Document {i+1} - Confidence: {score:.2f}]\n{text}"
                for i, (text, score) in enumerate(similar_docs)
            )
            
            # Prompt oluştur
            prompt = f"""Answer the following question based on the documents:
            
Question: {query}

Relevant Documents:
{context}

Reasoning Process:
1. Analyze each document's relevance to the question.
2. Identify key information from the most relevant documents.
3. Synthesize information into a coherent answer.

Answer:"""
            return prompt
        except Exception as e:
            # Hata durumunda basit bir prompt döndür
            print(f"Prompt oluşturulurken bir hata oluştu: {str(e)}")
            return f"Answer the following question: {query}"