import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import ollama
import tempfile
import os
import time
from pathlib import Path
import logging
import requests
from io import BytesIO
import arxiv

# ---------------------------
# Logging configuration
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------
# Page configuration and CSS
# ---------------------------
st.set_page_config(
    page_title="Advanced RAG System with ArXiv Integration",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #424242;
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: #E3F2FD;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .success-box {
            background-color: #E8F5E9;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .warning-box {
            background-color: #FFF8E1;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='main-header'>📚 Advanced RAG System with ArXiv Integration</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='info-box'>Bu uygulama PDF dosyalarını yüklemenize ya da ArXiv’den araştırma makaleleri getirmenize olanak tanır. Her iki akış için ayrı soru-cevap (QA) işlevselliği mevcuttur.</div>",
    unsafe_allow_html=True,
)

# ---------------------------
# Persistent directory for Chroma vector store
# ---------------------------
PERSIST_DIRECTORY = Path("./chroma_db")
PERSIST_DIRECTORY.mkdir(exist_ok=True)

# ---------------------------
# Helper: Safe rerun function
# ---------------------------
def safe_rerun():
    """Streamlit experimental_rerun fonksiyonunu güvenli şekilde çağırır."""
    if hasattr(st, "experimental_rerun"):
        try:
            st.experimental_rerun()
        except Exception as e:
            st.warning("Yeniden yükleme yapılamadı. Lütfen sayfayı manuel olarak yenileyin.")
    else:
        st.warning("Kullandığınız Streamlit sürümü yeniden yüklemeyi desteklemiyor. Lütfen sayfayı manuel olarak yenileyin.")

# ---------------------------
# Sidebar Settings
# ---------------------------
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚙️ System Settings</h2>", unsafe_allow_html=True)

    with st.expander("📊 Text Splitting Settings", expanded=True):
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000,
                               help="Daha büyük chunk'lar daha fazla bağlam sağlar ancak gereksiz bilgi de içerebilir.")
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200,
                                  help="Bilgi kaybını önlemek için chunk'lar arasında örtüşme.")

    with st.expander("🤖 Model Settings", expanded=True):
        model_name = st.selectbox(
            "LLM Model",
            ["granite4:tiny-h", "qwen2.5:7b", "deepscaler:latest", "smollm2:latest"],
            help="Soru-cevap için dil modeli.",
        )
        embedding_model = st.selectbox(
            "Embedding Model",
            ["embeddinggemma:latest", "nomic-embed-text:latest", "mxbai-embed-large"],
            help="Metin gömme modeli.",
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.1,
                                help="Düşük değerler daha deterministik, yüksek değerler daha yaratıcı sonuçlar üretir.")

    with st.expander("🔍 Search Settings", expanded=True):
        top_k = st.slider("Number of Contexts", 1, 10, 4,
                          help="Cevap üretirken kullanılacak ilgili belge parçalarının sayısı.")
        use_compression = st.checkbox("Use Context Compression", True,
                                      help="Bağlam sıkıştırması için LLM tabanlı özetleme kullanır (şimdilik uygulanmıyor).")

    if st.button("💾 Save Settings", use_container_width=True):
        st.success("Settings saved!")

    st.markdown("---")
    st.markdown("**About this app:**")
    st.info("Bu RAG sistemi, yerel Ollama modelleri kullanarak belge ve ArXiv makaleleri üzerinde akıllı soru-cevap gerçekleştirir.")

# ---------------------------
# Helper Functions
# ---------------------------
def load_and_process_pdf(uploaded_file, chunk_size, chunk_overlap):
    """
    PDF dosyasını geçici bir dosyaya yazar, PyMuPDFLoader ile okur,
    metni parçalar ve chunk'lar oluşturur.
    """
    try:
        original_name = uploaded_file.name
        # Geçici dosya oluşturma
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        logger.info(f"Loading PDF: {original_name}")
        loader = PyMuPDFLoader(tmp_file_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata["source"] = original_name

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        texts = text_splitter.split_documents(documents)
        logger.info(f"Created {len(texts)} text chunks from {original_name}")

        # Geçici dosyayı sil (hata olursa log'la)
        try:
            os.unlink(tmp_file_path)
        except Exception as unlink_error:
            logger.warning(f"Temporary file deletion failed: {unlink_error}")

        return texts
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        st.error(f"Error processing PDF: {str(e)}")
        return None

def create_vector_store(texts, embeddings):
    try:
        logger.info("Creating vector store")
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=str(PERSIST_DIRECTORY),
        )
        logger.info("Vector store created")
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        st.error(f"Error creating vector store: {str(e)}")
        return None

def create_qa_chain(llm, retriever):
    try:
        logger.info("Creating QA chain")
        template = """[INST] <<SYS>>
Answer the question using the provided context. Cite your sources.
If you don't know the answer, say "I don't know."
<</SYS>>

Context: {context}

Question: {question} 

Detailed Answer: [/INST]"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
        return qa_chain
    except Exception as e:
        logger.error(f"Error creating QA chain: {e}")
        st.error(f"Error creating QA chain: {str(e)}")
        return None

# ---------------------------
# Session State Initialization
# ---------------------------
if "upload_pdf_files" not in st.session_state:
    st.session_state["upload_pdf_files"] = []
if "arxiv_pdf_files" not in st.session_state:
    st.session_state["arxiv_pdf_files"] = []

# ---------------------------
# Tabs: Upload PDF and ArXiv Papers
# ---------------------------
tab_upload, tab_arxiv = st.tabs(["📤 Upload PDF", "📋 ArXiv Papers"])

# ========= Upload PDF Tab =========
with tab_upload:
    st.markdown("### Upload PDF Files")
    uploaded_files = st.file_uploader(
        "PDF dosyalarını yükleyin",
        type="pdf",
        accept_multiple_files=True,
        help="Birden fazla PDF dosyası yükleyebilirsiniz.",
        key="uploader_upload"
    )
    if uploaded_files:
        st.session_state["upload_pdf_files"] = uploaded_files

    if st.session_state["upload_pdf_files"]:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### PDF İşleme ve Soru-Cevap")
        all_texts = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        files = st.session_state["upload_pdf_files"]

        for i, file in enumerate(files):
            status_text.text(f"Processing PDF: {file.name} ({i+1}/{len(files)})")
            texts = load_and_process_pdf(file, chunk_size, chunk_overlap)
            if texts:
                all_texts.extend(texts)
            progress_bar.progress((i + 1) / len(files))
        progress_bar.empty()
        status_text.empty()

        if all_texts:
            st.markdown("<div class='success-box'>✅ PDF files processed successfully!</div>", unsafe_allow_html=True)
            st.write(f"Total {len(all_texts)} text chunks created.")

            # Embeddings ve vector store oluşturma
            with st.spinner("Creating vector store..."):
                embeddings = OllamaEmbeddings(model=embedding_model)
                vectorstore = create_vector_store(all_texts, embeddings)

            if vectorstore:
                llm = ChatOllama(model=model_name, temperature=temperature)
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
                qa_chain = create_qa_chain(llm, retriever)

                st.markdown("### 🤔 Ask a Question")
                query = st.text_input(
                    "Belge hakkında bir soru sorun:",
                    key="query_upload",
                    placeholder="Örneğin, bu belgede ana konular nelerdir?"
                )
                if query:
                    start_time = time.time()
                    with st.spinner("Generating answer..."):
                        try:
                            result = qa_chain({"query": query})
                            elapsed_time = time.time() - start_time
                            st.markdown("#### 📝 Answer:")
                            st.markdown(f"<div style='background-color:#f0f8ff;padding:1rem;border-radius:0.5rem;'>{result['result']}</div>", unsafe_allow_html=True)
                            if "source_documents" in result and result["source_documents"]:
                                st.markdown("#### 📚 Sources:")
                                for i, doc in enumerate(result["source_documents"], 1):
                                    source = doc.metadata.get("source", "Unknown")
                                    page = doc.metadata.get("page", 0)
                                    st.markdown(f"**Source {i}:** {source} (Page {page+1})")
                            st.info(f"Answer generated in {elapsed_time:.2f} seconds")
                        except Exception as e:
                            logger.error(f"Error generating answer: {e}")
                            st.error(f"Error generating answer: {str(e)}")
        else:
            st.info("Yüklenen PDF dosyalarından herhangi bir metin çıkarılamadı.")

    if st.button("Clear Uploaded PDFs", key="clear_upload"):
        st.session_state["upload_pdf_files"] = []
        safe_rerun()

# ========= ArXiv Papers Tab =========
with tab_arxiv:
    st.markdown("### Search and Analyze ArXiv Papers")
    arxiv_query = st.text_input("Search ArXiv (use English keywords)", key="arxiv_search")
    selected_paper = None

    if arxiv_query:
        try:
            search = arxiv.Search(
                query=arxiv_query,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            papers = []
            for result in search.results():
                papers.append({
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "published": result.published.strftime("%Y-%m-%d"),
                    "pdf_url": result.pdf_url,
                    "summary": result.summary,
                })

            if papers:
                paper_titles = [f"{p['title']} ({p['published']})" for p in papers]
                selected_title = st.selectbox("Select a paper", paper_titles)
                selected_paper = papers[paper_titles.index(selected_title)]

                if selected_paper:
                    with st.expander("Paper Details"):
                        st.markdown(f"**Title:** {selected_paper['title']}")
                        st.markdown(f"**Authors:** {', '.join(selected_paper['authors'])}")
                        st.markdown(f"**Summary:** {selected_paper['summary']}")

                    if st.button("Download and Process Paper", key="download_arxiv"):
                        with st.spinner("Downloading paper..."):
                            response = requests.get(selected_paper["pdf_url"])
                            if response.status_code == 200:
                                downloaded_file = BytesIO(response.content)
                                downloaded_file.name = f"arXiv_{selected_paper['title'][:50]}.pdf"
                                st.session_state["arxiv_pdf_files"] = [downloaded_file]
                                safe_rerun()
                            else:
                                st.error("Failed to download the paper!")
            else:
                st.warning("No papers found for your query.")
        except Exception as e:
            st.error(f"Error during ArXiv search: {str(e)}")

    if st.session_state["arxiv_pdf_files"]:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### PDF Processing and Q&A for ArXiv Paper")
        all_texts = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        files = st.session_state["arxiv_pdf_files"]

        for i, file in enumerate(files):
            status_text.text(f"Processing PDF: {file.name} ({i+1}/{len(files)})")
            texts = load_and_process_pdf(file, chunk_size, chunk_overlap)
            if texts:
                all_texts.extend(texts)
            progress_bar.progress((i + 1) / len(files))
        progress_bar.empty()
        status_text.empty()

        if all_texts:
            st.markdown("<div class='success-box'>✅ PDF files processed successfully!</div>", unsafe_allow_html=True)
            st.write(f"Total {len(all_texts)} text chunks created.")

            with st.spinner("Creating vector store..."):
                embeddings = OllamaEmbeddings(model=embedding_model)
                vectorstore = create_vector_store(all_texts, embeddings)

            if vectorstore:
                llm = ChatOllama(model=model_name, temperature=temperature)
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
                qa_chain = create_qa_chain(llm, retriever)

                st.markdown("### 🤔 Ask a Question")
                query = st.text_input(
                    "Ask a question about the document:",
                    key="query_arxiv",
                    placeholder="E.g., What are the main topics in this document?"
                )
                if query:
                    start_time = time.time()
                    with st.spinner("Generating answer..."):
                        try:
                            result = qa_chain({"query": query})
                            elapsed_time = time.time() - start_time
                            st.markdown("#### 📝 Answer:")
                            st.markdown(f"<div style='background-color:#f0f8ff;padding:1rem;border-radius:0.5rem;'>{result['result']}</div>", unsafe_allow_html=True)
                            if "source_documents" in result and result["source_documents"]:
                                st.markdown("#### 📚 Sources:")
                                for i, doc in enumerate(result["source_documents"], 1):
                                    source = doc.metadata.get("source", "Unknown")
                                    page = doc.metadata.get("page", 0)
                                    st.markdown(f"**Source {i}:** {source} (Page {page+1})")
                            st.info(f"Answer generated in {elapsed_time:.2f} seconds")
                        except Exception as e:
                            logger.error(f"Error generating answer: {e}")
                            st.error(f"Error generating answer: {str(e)}")
        else:
            st.info("No text chunks could be created from the downloaded paper.")

    if st.button("Clear ArXiv PDFs", key="clear_arxiv"):
        st.session_state["arxiv_pdf_files"] = []
        safe_rerun()

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; opacity: 0.7; font-size: 0.8rem;">
        Advanced RAG System | Powered by LangChain + Ollama + Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
