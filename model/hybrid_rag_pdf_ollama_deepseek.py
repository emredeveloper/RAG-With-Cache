import streamlit as st
import os
import time
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import base64

from langchain_community.document_loaders import PDFPlumberLoader, PyMuPDFLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.retrievers import BM25Retriever
from nltk.tokenize import word_tokenize
from langchain.retrievers import EnsembleRetriever

template = """
You are an expert assistant for question-answering tasks. You must follow these rules strictly:
1. ONLY answer based on the context provided below - do not use any other knowledge.
2. If the answer is not clearly in the context, say "Based on the provided context, I don't have enough information to answer that question."
3. Be precise and concise - use factual information from the context only.
4. Do not hallucinate or make up information that isn't in the context.
5. Format your answer in a clear, straightforward manner.
6. DO NOT include citations, reference numbers, or footnotes like [1], [2], etc.
7. DO NOT start your answer with phrases like "Based on the context" or "According to the document".
8. Just provide a direct, factual answer based on the context.

Question: {question} 

Context:
{context}

Answer (responding ONLY with information from the context):
"""

pdfs_directory = 'hybrid-retrieval-rag/pdfs/'

model = OllamaLLM(
    model="granite4:tiny-h",
    temperature=0.1,  # Lower temperature for more deterministic responses
    top_p=0.9,        # More focused sampling
    num_ctx=4096      # Ensure sufficient context window
)

def ensure_directory_exists(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def upload_pdf(file):
    ensure_directory_exists(pdfs_directory)  # Ensure directory exists before saving
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def analyze_pdf_structure(file_path):
    """Analyze PDF structure and return statistics"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        stats = {
            "Page count": len(doc),
            "Average text length": 0,
            "Pages with text": 0,
            "Pages with images": 0
        }
        
        total_text = 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if len(text.strip()) > 0:
                stats["Pages with text"] += 1
                total_text += len(text)
            
            if len(page.get_images()) > 0:
                stats["Pages with images"] += 1
        
        if stats["Pages with text"] > 0:
            stats["Average text length"] = total_text / stats["Pages with text"]
        
        # Get a thumbnail of first page
        first_page = doc[0]
        pix = first_page.get_pixmap(matrix=fitz.Matrix(0.2, 0.2))
        img_bytes = pix.tobytes("png")
        
        doc.close()
        return stats, img_bytes
    except Exception as e:
        return {"Error": str(e)}, None

def load_pdf(file_path, loader_type="PyMuPDF"):
    """Load PDF with selected loader"""
    with st.spinner(f'Loading PDF using {loader_type}...'):
        if loader_type == "PyMuPDF":
            loader = PyMuPDFLoader(file_path)
        elif loader_type == "PyPDF":
            loader = PyPDFLoader(file_path)
        elif loader_type == "PDFPlumber":
            loader = PDFPlumberLoader(file_path)
        else:
            st.error(f"Unknown loader type: {loader_type}")
            return []
            
        documents = loader.load()
        st.success(f"Loaded {len(documents)} pages from PDF")
        return documents

def split_text(documents, chunk_size=1000, chunk_overlap=200, splitter_type="Recursive"):
    """Split text with configurable parameters and splitter type"""
    with st.spinner(f'Processing document content with {splitter_type} splitter...'):
        if splitter_type == "Recursive":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True
            )
        else:  # "Character"
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True
            )
            
        chunks = text_splitter.split_documents(documents)
        st.success(f"Created {len(chunks)} text chunks")
        return chunks

def visualize_chunks(documents, chunks):
    """Visualize how the text is divided into chunks"""
    st.subheader("Chunk Distribution Visualization")
    
    # Calculate total document length and where chunks start/end
    total_text = ""
    page_lengths = []
    page_titles = []
    
    for i, doc in enumerate(documents):
        page_text = doc.page_content
        total_text += page_text
        page_lengths.append(len(page_text))
        page_titles.append(f"Page {i+1}")
    
    # Create figure for visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot pages as segments
    start = 0
    page_positions = [0]
    for length in page_lengths:
        ax.axvline(x=start + length, color='r', linestyle='--', alpha=0.7)
        start += length
        page_positions.append(start)
    
    # Plot chunks as segments
    chunk_starts = []
    chunk_ends = []
    for chunk in chunks:
        text = chunk.page_content
        start_idx = total_text.find(text)
        if start_idx != -1:
            chunk_starts.append(start_idx)
            chunk_ends.append(start_idx + len(text))
    
    # Plot chunks as horizontal lines
    for i, (start, end) in enumerate(zip(chunk_starts, chunk_ends)):
        ax.plot([start, end], [i, i], 'b-', linewidth=2)
    
    ax.set_yticks(range(len(chunks)))
    ax.set_yticklabels([f"Chunk {i+1}" for i in range(len(chunks))])
    ax.set_xlabel("Character Position")
    ax.set_title("Document Chunking Visualization")
    
    # Add page markers
    for i, pos in enumerate(page_positions[:-1]):
        ax.text(pos + page_lengths[i]/2, len(chunks) + 1, page_titles[i], 
                ha='center', va='center', fontsize=8, color='r')
    
    plt.tight_layout()
    st.pyplot(fig)

def display_pdf_preview(file_path):
    try:
        # Try to use PyMuPDF to render PDF pages as images instead of iframe
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        
        st.write(f"PDF has {len(doc)} pages")
        
        # Show first 5 pages only to avoid performance issues
        pages_to_display = min(5, len(doc))
        st.write(f"Showing first {pages_to_display} pages")
        
        for page_num in range(pages_to_display):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_bytes = pix.tobytes("png")
            
            st.subheader(f"Page {page_num + 1}")
            # Replace deprecated use_column_width with use_container_width
            st.image(Image.open(io.BytesIO(img_bytes)), use_container_width=True)
        
        # Offer download button for the full PDF
        with open(file_path, "rb") as file:
            st.download_button(
                label="Download PDF",
                data=file,
                file_name=os.path.basename(file_path),
                mime="application/pdf"
            )
    except Exception as e:
        st.error(f"Error displaying PDF preview: {str(e)}")
        # Offer an alternative download option
        with open(file_path, "rb") as file:
            st.download_button(
                label="Download PDF to view externally",
                data=file,
                file_name=os.path.basename(file_path),
                mime="application/pdf"
            )

def build_semantic_retriever(documents):
    with st.spinner('Building semantic index...'):
        embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(documents)
        return vector_store.as_retriever(search_kwargs={"k": 5})  # Increased k for better coverage

def build_bm25_retriever(documents):
    with st.spinner('Building keyword index...'):
        return BM25Retriever.from_documents(documents, preprocess_func=word_tokenize, k=5)  # Increased k

def clean_model_output(text):
    """Remove <think> blocks and other unwanted patterns from model output"""
    # Remove <think> sections
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Remove any numbered references like [1], [2], etc.
    cleaned_text = re.sub(r'\[\d+\]', '', cleaned_text)
    
    # Remove multiple newlines and clean up spacing
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()

def answer_question(question, documents, show_reasoning=False):
    with st.spinner('Generating answer...'):
        # Format context with clear section breaks and metadata
        formatted_chunks = []
        for i, doc in enumerate(documents):
            # Add page number if available
            page_info = f"Page {doc.metadata.get('page', 'unknown')}" if hasattr(doc, 'metadata') and doc.metadata else ""
            # Add a clear chunk identifier and content
            chunk_text = f"CHUNK {i+1} {page_info}:\n{doc.page_content}\n"
            formatted_chunks.append(chunk_text)
        
        # Join with clear separators
        context = "\n" + "-"*40 + "\n".join(formatted_chunks) + "\n" + "-"*40 + "\n"
        
        # Show reasoning process if enabled
        if show_reasoning:
            st.subheader("🧠 Reasoning Process")
            st.write("Context being analyzed:")
            with st.expander("Context sent to model"):
                st.text(context)
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        
        try:
            # Generate answer
            response = chain.invoke({"question": question, "context": context})
            
            # Clean the response
            cleaned_response = clean_model_output(response)
            return cleaned_response
        except Exception as e:
            st.error(f"Error during answer generation: {str(e)}")
            return "I encountered an error while processing your question. Please try again with a different question."

st.title("PDF Question Answering with DeepSeek")

# Create sidebar for settings and debug
with st.sidebar:
    st.header("Settings")
    
    # PDF loader settings
    pdf_loader = st.selectbox(
        "PDF Loader", 
        ["PyMuPDF", "PyPDF", "PDFPlumber"],
        index=0,
        help="PyMuPDF usually has the best text extraction quality"
    )
    
    # Chunking settings
    st.subheader("Chunking Settings")
    chunk_size = st.slider("Chunk Size", 200, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
    splitter_type = st.selectbox(
        "Text Splitter", 
        ["Recursive", "Character"],
        index=0
    )
    
    # Display settings
    st.subheader("Display Settings")
    show_chunks = st.checkbox("Show retrieved text chunks", value=True)
    show_pdf_content = st.checkbox("Show extracted text content", value=False)
    show_pdf_preview = st.checkbox("Show PDF preview", value=False)
    show_chunk_viz = st.checkbox("Show chunk visualization", value=False)
    
    # Add Show Reasoning toggle
    st.subheader("Advanced Settings")
    show_reasoning = st.checkbox("Show model reasoning process", value=False)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

# Fix the PDF content display to avoid nested expanders
if uploaded_file:
    upload_pdf(uploaded_file)
    pdf_path = pdfs_directory + uploaded_file.name
    
    # Analyze PDF structure
    with st.expander("📊 PDF Structure Analysis"):
        stats, thumbnail = analyze_pdf_structure(pdf_path)
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if thumbnail:
                st.image(Image.open(io.BytesIO(thumbnail)), caption="First Page", use_container_width=True)
        
        with col2:
            for key, value in stats.items():
                if key == "Average text length":
                    st.write(f"**{key}:** {value:.1f} characters")
                else:
                    st.write(f"**{key}:** {value}")
            
            recommended_chunk = min(1000, max(200, int(stats.get("Average text length", 1000) / 2)))
            st.write(f"**Recommended chunk size:** ~{recommended_chunk} characters")
    
    # Show PDF preview if enabled
    if show_pdf_preview:
        with st.expander("📄 PDF Preview", expanded=True):
            display_pdf_preview(pdf_path)
    
    # Load PDF with selected loader
    documents = load_pdf(pdf_path, loader_type=pdf_loader)
    
    # Option to display extracted PDF content - FIXED to avoid nested expanders
    if show_pdf_content:
        st.subheader("🔤 Extracted PDF Text")
        # Use a page selector instead of nested expanders
        page_numbers = [f"Page {i+1}" for i in range(len(documents))]
        if page_numbers:
            # Initialize session state for page selection if needed
            if "page_selection" not in st.session_state:
                st.session_state["page_selection"] = page_numbers[0]
                
            selected_page = st.selectbox("Select page to view", 
                                         page_numbers, 
                                         index=page_numbers.index(st.session_state.get("page_selection", page_numbers[0])))
            page_idx = int(selected_page.split(" ")[1]) - 1
            st.text_area("Page Content", documents[page_idx].page_content, height=300)
            st.write(f"Characters: {len(documents[page_idx].page_content)}")
            
            # Add page navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if page_idx > 0:
                    if st.button("◀️ Previous Page"):
                        # Use st.rerun() instead of experimental_rerun
                        st.session_state["page_selection"] = page_numbers[page_idx - 1]
                        st.rerun()
            with col2:
                if page_idx < len(documents) - 1:
                    if st.button("Next Page ▶️"):
                        # Use st.rerun() instead of experimental_rerun
                        st.session_state["page_selection"] = page_numbers[page_idx + 1]
                        st.rerun()
    
    # Split documents into chunks with selected parameters
    chunked_documents = split_text(documents, chunk_size, chunk_overlap, splitter_type)
    
    # Show chunk visualization if enabled
    if show_chunk_viz and chunked_documents:
        with st.expander("📊 Chunk Visualization", expanded=False):
            visualize_chunks(documents, chunked_documents)
    
    # Create retrievers
    semantic_retriever = build_semantic_retriever(chunked_documents)
    bm25_retriever = build_bm25_retriever(chunked_documents)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.8, 0.2]  # Give more weight to semantic search
    )
    
    st.subheader("Ask a question about the PDF")
    question = st.chat_input("Type your question here")

    if question:
        st.chat_message("user").write(question)
        
        # Get relevant documents
        related_documents = hybrid_retriever.invoke(question)
        
        # Display a summary of what was retrieved
        st.info(f"Found {len(related_documents)} relevant chunks from the PDF")
        
        # Show retrieved chunks if enabled
        if show_chunks:
            with st.expander("Retrieved Text Chunks", expanded=True):
                for i, doc in enumerate(related_documents):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.text(doc.page_content)
                    st.divider()
        
        try:
            answer = answer_question(question, related_documents, show_reasoning=show_reasoning)
            message_placeholder = st.chat_message("assistant")
            message_placeholder.write(answer)
            
            # Add feedback buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Good Answer"):
                    st.success("Thank you for your feedback!")
            with col2:
                if st.button("👎 Bad Answer"):
                    st.error("We'll try to improve!")
                    
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            st.info("Try rephrasing your question or uploading a different PDF.")

