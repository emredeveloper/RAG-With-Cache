import streamlit as st
import os
import time
import re

# Make matplotlib import optional due to NumPy compatibility issues
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    st.warning("⚠️ Matplotlib not available due to NumPy compatibility issues. Chunk visualization will be disabled.")
    MATPLOTLIB_AVAILABLE = False
    plt = None
    np = None

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

# DuckDuckGo search (optional)
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except Exception:
    DDGS_AVAILABLE = False

# Import Chonkie chunkers
# Import Chonkie chunkers safely; fall back if not installed
try:
    from chonkie import TokenChunker, TableChunker, SemanticChunker, SentenceChunker, RecursiveChunker
    CHONKIE_AVAILABLE = True
except Exception:
    CHONKIE_AVAILABLE = False


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

# Turkish template for Turkish questions
template_tr = """
Sen bir soru-cevap asistanısın. Aşağıdaki kurallara kesinlikle uymalısın:
1. Yalnızca aşağıdaki bağlamdan cevap ver; başka bilgi kullanma.
2. Eğer cevap bağlamda açıkça yoksa, tam olarak şu cümleyi yaz: "Verilen bağlam doğrultusunda bu soruya cevap verecek yeterli bilgiye sahip değilim."
3. Kısa ve net ol; sadece bağlamdaki bilgiyi kullan.
4. Uydurma yapma.
5. Cevabında referans numarası, dipnot veya kaynak işareti kullanma.
6. Cevabı doğrudan ver, gereksiz giriş cümleleri kullanma.

Soru: {question}

Bağlam:
{context}

Cevap (yalnızca bağlamdan doğrulanabilen bilgiyi ver):
"""

def is_turkish(text: str) -> bool:
    """Heuristic to detect if the text is Turkish.

    Uses presence of Turkish characters and common Turkish words.
    """
    if not text:
        return False
    import re
    # check for Turkish-specific characters or common Turkish question words
    if re.search('[ışğüçİŞĞÜÇ]', text):
        return True
    if re.search(r'\b(nedir|ne|neden|nasıl|kim|nerede|şu|bu|hangi|kaç)\b', text.lower()):
        return True
    return False

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
        # Combine all document content for heuristics
        combined_text = ""
        page_texts = []
        for doc in documents:
            page_texts.append(doc.page_content)
            combined_text += doc.page_content + "\n\n"

        # Auto-selection: if splitter_type is 'Auto', pick best available chunker
        chosen_chunks = []
        use_chonkie = CHONKIE_AVAILABLE

        def looks_like_table(text):
            # quick heuristics: pipes, multiple consecutive dashes, or table-like rows
            return ('|' in text and any(line.count('|') > 1 for line in text.splitlines())) or ('\t' in text)

        # If user requested Auto, choose automatically
        if splitter_type == "Auto":
            # If document contains tables, prefer TableChunker
            table_like = any(looks_like_table(p) for p in page_texts)
            avg_len = sum(len(p) for p in page_texts) / max(1, len(page_texts))
            short_lines_ratio = sum(1 for p in page_texts for l in p.splitlines() if len(l.strip()) < 40) / max(1, sum(len(p.splitlines()) for p in page_texts))

            if table_like and use_chonkie:
                try:
                    table_chunker = TableChunker()
                    for p in page_texts:
                        chunks = table_chunker.chunk(p)
                        chosen_chunks.extend([c.text for c in chunks])
                    st.info("Using TableChunker (Chonkie) for tabular content")
                    st.success(f"Created {len(chosen_chunks)} text chunks")
                    return chosen_chunks
                except Exception:
                    use_chonkie = False

            # If long documents with structure: use RecursiveChunker (Chonkie) or RecursiveCharacterTextSplitter
            if avg_len > 3000 and use_chonkie:
                try:
                    recursive = RecursiveChunker()
                    for p in page_texts:
                        chunks = recursive.chunk(p)
                        chosen_chunks.extend([c.text for c in chunks])
                    st.info("Using RecursiveChunker (Chonkie) for long structured documents")
                    st.success(f"Created {len(chosen_chunks)} text chunks")
                    return chosen_chunks
                except Exception:
                    use_chonkie = False

            # If semantic chunking available and content is topically coherent, prefer SemanticChunker
            if use_chonkie:
                try:
                    semantic = SemanticChunker()
                    for p in page_texts:
                        chunks = semantic.chunk(p)
                        chosen_chunks.extend([c.text for c in chunks])
                    if chosen_chunks:
                        st.info("Using SemanticChunker (Chonkie) for topical coherence")
                        st.success(f"Created {len(chosen_chunks)} text chunks")
                        return chosen_chunks
                except Exception:
                    pass

            # Heuristic: if many short lines (likely slides/OCR), use TokenChunker for stable sizes
            if short_lines_ratio > 0.35 and use_chonkie:
                try:
                    token_chunker = TokenChunker()
                    for p in page_texts:
                        chunks = token_chunker.chunk(p)
                        chosen_chunks.extend([c.text for c in chunks])
                    st.info("Using TokenChunker (Chonkie) for short-line / OCR-like content")
                    st.success(f"Created {len(chosen_chunks)} text chunks")
                    return chosen_chunks
                except Exception:
                    pass

            # Fallback to LangChain Recursive or Character splitters depending on avg length
            if avg_len > 2000:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
            else:
                text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)

            chosen_chunks = [doc.page_content for doc in text_splitter.split_documents(documents)]
            st.success(f"Created {len(chosen_chunks)} text chunks (langchain fallback)")
            return chosen_chunks

        # If not Auto, respect user choice between 'Recursive' and 'Character'
        if splitter_type == "Recursive":
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
        else:  # 'Character'
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)

        chunks = [doc.page_content for doc in text_splitter.split_documents(documents)]
        st.success(f"Created {len(chunks)} text chunks")
        return chunks

def visualize_chunks(documents, chunks):
    """Visualize how the text is divided into chunks"""
    if not MATPLOTLIB_AVAILABLE:
        st.warning("Chunk visualization is not available due to matplotlib import issues.")
        return
        
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
        # Fix: chunks are now strings, not Document objects
        text = chunk  # chunk is already a string
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
            # Update deprecated use_container_width to width='stretch'
            st.image(Image.open(io.BytesIO(img_bytes)), width='stretch')
        
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

def build_semantic_retriever(chunks):
    with st.spinner('Building semantic index...'):
        embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
        vector_store = InMemoryVectorStore(embeddings)
        # Create documents from chunks for the vector store
        from langchain_core.documents import Document
        documents = [Document(page_content=chunk) for chunk in chunks]
        vector_store.add_documents(documents)
        return vector_store.as_retriever(search_kwargs={"k": 5})

def build_bm25_retriever(chunks):
    with st.spinner('Building keyword index...'):
        from langchain_core.documents import Document
        documents = [Document(page_content=chunk) for chunk in chunks]
        return BM25Retriever.from_documents(documents, preprocess_func=word_tokenize, k=5)

def get_web_content(query, max_results=5):
    """Fetch web snippets using DuckDuckGo (ddgs). Returns combined text or an error message."""
    if not DDGS_AVAILABLE:
        return "Web search not available (ddgs package not installed)."

    try:
        all_results = []
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            if results:
                all_results.extend(results)

        if not all_results:
            return "No web content found."

        web_content = []
        for res in all_results[:max_results]:
            title = res.get('title', 'No title')
            body = res.get('body', '')
            url = res.get('url', '')
            snippet = body[:500] + '...' if body and len(body) > 500 else body
            web_content.append(f"[WEB: {url}] {title}\n{snippet}")

        return "\n\n".join(web_content)
    except Exception as e:
        return f"Web search failed: {e}"

def clean_model_output(text):
    """Remove <think> blocks and other unwanted patterns from model output"""
    # Remove <think> sections
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Remove any numbered references like [1], [2], etc.
    cleaned_text = re.sub(r'\[\d+\]', '', cleaned_text)
    
    # Remove multiple newlines and clean up spacing
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()


def llm_rerank(query, docs, top_k=5):
    """Simple LLM-based reranker: asks the model to score each doc for relevance.

    Returns the docs sorted by descending relevance (top_k returned).
    """
    # Build a short scoring prompt
    prompt_parts = [
        "You are a relevance rater. Score each document for relevance to the query on a scale 0-100.",
        f"Query: {query}",
        "\nDocuments:\n"
    ]

    for i, d in enumerate(docs):
        snippet = getattr(d, 'page_content', '')[:800]
        prompt_parts.append(f"[{i}] {snippet}\n")

    prompt_parts.append("\nRespond with lines of the form: index:score (e.g. 0:85). Only provide numeric scores for the given indices.")
    scoring_prompt = "\n".join(prompt_parts)

    try:
        prompt = ChatPromptTemplate.from_template(scoring_prompt)
        chain = prompt | model
        response = chain.invoke({})
        text = response if isinstance(response, str) else str(response)
    except Exception:
        # If scoring fails, return original docs
        return docs[:top_k]

    # Parse scores
    scores = {}
    import re
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s*[:\-]\s*(\d{1,3})", line)
        if m:
            idx = int(m.group(1))
            score = int(m.group(2))
            scores[idx] = score

    # Fallback: if no scores parsed, return original
    if not scores:
        return docs[:top_k]

    # Attach score to docs
    scored = []
    for i, d in enumerate(docs):
        s = scores.get(i, 0)
        scored.append((s, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored][:top_k]

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
        
        # Choose template based on question language
        use_template = template_tr if is_turkish(question) else template
        prompt = ChatPromptTemplate.from_template(use_template)
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

def extract_images_from_pdf(file_path, output_dir="extracted_images"):
    """Extract images from PDF and return image metadata"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        images_info = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Save image
                image_filename = f"page_{page_num+1}_img_{img_index+1}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Store image info
                images_info.append({
                    "page": page_num + 1,
                    "filename": image_filename,
                    "path": image_path,
                    "ext": image_ext,
                    "bbox": img[1:5] if len(img) > 4 else None  # bounding box
                })
        
        doc.close()
        return images_info
        
    except Exception as e:
        st.error(f"Error extracting images: {str(e)}")
        return []


def extract_images_from_pdf_with_screenshots(file_path, output_dir="extracted_images", render_page_images=True):
    """Extract embedded images and optionally render page screenshots for pages without images.

    Returns a list of image metadata dicts. Screenshots have key 'screenshot': True.
    """
    # Start by extracting embedded images
    images_info = extract_images_from_pdf(file_path, output_dir)

    # Ensure output dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not render_page_images:
        return images_info

    try:
        import fitz
        doc = fitz.open(file_path)

        # Determine which pages already have images
        pages_with_images = {info['page'] for info in images_info}

        for page_num in range(len(doc)):
            if (page_num + 1) in pages_with_images:
                continue

            page = doc[page_num]
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image_bytes = pix.tobytes("png")
                image_filename = f"page_{page_num+1}_screenshot.png"
                image_path = os.path.join(output_dir, image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                images_info.append({
                    "page": page_num + 1,
                    "filename": image_filename,
                    "path": image_path,
                    "ext": "png",
                    "screenshot": True,
                    "bbox": None
                })
            except Exception:
                # skip pages that fail to render
                continue

        doc.close()
    except Exception:
        # If PyMuPDF isn't available or rendering fails, just return embedded images
        pass

    return images_info

def display_relevant_images(images_info, answer_text, question):
    """Display images that are relevant to the answer"""
    if not images_info:
        return
    
    # Simple keyword matching to find relevant images
    relevant_keywords = [
        "figure", "fig", "table", "chart", "graph", "diagram", "image", "plot",
        "tablo", "grafik", "şekil", "çizelge", "resim"  # Turkish keywords
    ]
    
    # Check if answer mentions images or if question is about visual content
    answer_lower = answer_text.lower()
    question_lower = question.lower()
    
    mentions_visual = any(keyword in answer_lower or keyword in question_lower for keyword in relevant_keywords)
    
    if mentions_visual or "görsel" in question_lower or "image" in question_lower:
        st.subheader("📸 Relevant Images from PDF")
        
        # Display all images for now (could be improved with better relevance detection)
        cols = st.columns(min(3, len(images_info)))
        
        for i, img_info in enumerate(images_info):
            col_idx = i % 3
            with cols[col_idx]:
                try:
                    image = Image.open(img_info["path"])
                    # Update deprecated use_container_width to width='stretch'
                    st.image(image, caption=f"Page {img_info['page']} - {img_info['filename']}", width='stretch')
                except Exception as e:
                    st.error(f"Error loading image {img_info['filename']}: {str(e)}")


def _capture_chunk_screenshots(pdf_path, related_documents, max_images=3, output_dir=None):
    """Capture cropped screenshots from pdf pages that cover the text in related_documents.

    Returns list of dicts with keys: page, filename, path, screenshot=True
    """
    screenshots = []
    try:
        import fitz
    except Exception:
        return screenshots

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(pdf_path) or '.', 'extracted_images')
    os.makedirs(output_dir, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return screenshots

    count = 0
    for i, doc_chunk in enumerate(related_documents):
        if count >= max_images:
            break

        text = getattr(doc_chunk, 'page_content', '')
        # Use a short snippet to search in page text (avoid extremely long searches)
        snippet = text.strip()[:200]
        page_num = None
        # If metadata provides page, prefer it
        if hasattr(doc_chunk, 'metadata') and doc_chunk.metadata:
            page_meta = doc_chunk.metadata.get('page')
            try:
                page_num = int(page_meta) - 1 if page_meta is not None else None
            except Exception:
                page_num = None

        found = False
        search_rects = []

        if page_num is not None and 0 <= page_num < len(doc):
            # Search on the specified page
            try:
                page = doc[page_num]
                # search_for returns list of rects; we try multiple smaller substrings if needed
                for sub in [snippet, snippet[:100], snippet[:50]]:
                    if not sub.strip():
                        continue
                    rects = page.search_for(sub)
                    if rects:
                        search_rects.extend(rects)
                        found = True
                        break
            except Exception:
                pass

        if not found:
            # Try searching across all pages
            for pidx in range(len(doc)):
                try:
                    page = doc[pidx]
                    for sub in [snippet, snippet[:100], snippet[:50]]:
                        if not sub.strip():
                            continue
                        rects = page.search_for(sub)
                        if rects:
                            page_num = pidx
                            search_rects.extend(rects)
                            found = True
                            break
                    if found:
                        break
                except Exception:
                    continue

        if not found or page_num is None:
            continue

        # Union rects into a bbox
        x0 = min(r.x0 for r in search_rects)
        y0 = min(r.y0 for r in search_rects)
        x1 = max(r.x1 for r in search_rects)
        y1 = max(r.y1 for r in search_rects)

        # Expand bbox a bit for context (in PDF points)
        pad_x = (x1 - x0) * 0.05 + 10
        pad_y = (y1 - y0) * 0.05 + 10
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(doc[page_num].rect.x1, x1 + pad_x)
        y1 = min(doc[page_num].rect.y1, y1 + pad_y)

        try:
            page = doc[page_num]
            # high resolution render
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            # scale rect coordinates
            rect = fitz.Rect(x0, y0, x1, y1)
            r = fitz.Rect(rect.x0 * mat.a, rect.y0 * mat.d, rect.x1 * mat.a, rect.y1 * mat.d)
            # crop the pixmap
            cropped = pix.crop([int(r.x0), int(r.y0), int(r.x1), int(r.y1)])
            out_name = f"page_{page_num+1}_relevant_{i+1}.png"
            out_path = os.path.join(output_dir, out_name)
            cropped.save(out_path)
            screenshots.append({
                'page': page_num + 1,
                'filename': out_name,
                'path': out_path,
                'ext': 'png',
                'screenshot': True
            })
            count += 1
        except Exception:
            continue

    try:
        doc.close()
    except Exception:
        pass

    return screenshots


def display_relevant_images(images_info, answer_text, question, pdf_path=None, related_documents=None):
    """Display images that are relevant to the answer.

    If `related_documents` and `pdf_path` are given, try to capture cropped screenshots
    that cover the text used by the model. Otherwise, fall back to embedded images.
    """
    # Try to create screenshots based on retrieved documents first
    screenshots = []
    if related_documents and pdf_path:
        try:
            screenshots = _capture_chunk_screenshots(pdf_path, related_documents, max_images=3)
        except Exception:
            screenshots = []

    imgs_to_show = screenshots if screenshots else images_info

    if not imgs_to_show:
        return

    st.subheader("📸 Relevant Images from PDF")
    cols = st.columns(min(3, len(imgs_to_show)))

    for i, img_info in enumerate(imgs_to_show):
        col_idx = i % 3
        with cols[col_idx]:
            try:
                image = Image.open(img_info["path"])
                caption = f"Page {img_info.get('page', '?')} - {img_info.get('filename', '')}"
                if img_info.get('screenshot'):
                    caption += " (screenshot)"
                st.image(image, caption=caption, width='stretch')
                # Add a download button for each image
                with open(img_info["path"], 'rb') as f:
                    st.download_button(label="Download image", data=f, file_name=img_info.get('filename', 'image.png'))
            except Exception as e:
                st.error(f"Error loading image {img_info.get('filename','')}: {str(e)}")

st.title("PDF Question Answering with DeepSeek")

# Create sidebar for minimal settings and debug
with st.sidebar:
    st.header("Settings")

    # Minimal display settings
    st.subheader("Display Settings")
    show_chunks = st.checkbox("Show retrieved text chunks", value=True)
    render_screenshots_by_default = st.checkbox(
        "Render page screenshots for pages without embedded images",
        value=False,
        help="When enabled, pages that don't contain embedded images will be rendered as images (useful for scanned PDFs)."
    )

    # Web search settings
    st.subheader("Web Search")
    if DDGS_AVAILABLE:
        include_web_by_default = st.checkbox("Include web results by default", value=False,
                                            help="When enabled, the app will fetch DuckDuckGo snippets for each query unless overridden per query.")
    else:
        include_web_by_default = False
        st.info("Web search disabled: install 'ddgs' to enable DuckDuckGo snippets.")

    # Add Show Reasoning toggle
    st.subheader("Advanced Settings")
    show_reasoning = st.checkbox("Show model reasoning process", value=False)
    # Reranker option
    st.subheader("Retrieval")
    enable_rerank = st.checkbox("Enable LLM reranker (slower, more accurate)", value=False)

uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

# Fix the PDF content display to avoid nested expanders
if uploaded_file:
    upload_pdf(uploaded_file)
    pdf_path = pdfs_directory + uploaded_file.name
    
    # Extract images from PDF
    images_dir = os.path.join(pdfs_directory, "extracted_images")
    # Use screenshot-capable extractor when requested
    try:
        images_info = extract_images_from_pdf_with_screenshots(pdf_path, images_dir, render_page_images=render_screenshots_by_default)
    except NameError:
        # Backwards compatible: fallback to original extractor if helper not defined
        images_info = extract_images_from_pdf(pdf_path, images_dir)
    
    if images_info:
        st.info(f"📸 Extracted {len(images_info)} images from PDF")
    
    # Analyze PDF structure
    with st.expander("📊 PDF Structure Analysis"):
        stats, thumbnail = analyze_pdf_structure(pdf_path)
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if thumbnail:
                # Update deprecated use_container_width to width='stretch'
                st.image(Image.open(io.BytesIO(thumbnail)), caption="First Page", width='stretch')
        
        with col2:
            for key, value in stats.items():
                if key == "Average text length":
                    st.write(f"**{key}:** {value:.1f} characters")
                else:
                    st.write(f"**{key}:** {value}")
            
            recommended_chunk = min(1000, max(200, int(stats.get("Average text length", 1000) / 2)))
            st.write(f"**Recommended chunk size:** ~{recommended_chunk} characters")
    
    # Load PDF with PyPDF loader only
    documents = load_pdf(pdf_path, loader_type="PyPDF")
    
    # Split documents into chunks using Auto mode (content-aware)
    chunked_documents = split_text(documents, splitter_type="Auto")
    
    # Create retrievers once and cache them in session state to avoid rebuilding on every rerun
    if "retriever_for" not in st.session_state or st.session_state.get("retriever_for") != pdf_path:
        with st.spinner('Building retrievers for this PDF...'):
            semantic_retriever = build_semantic_retriever(chunked_documents)
            bm25_retriever = build_bm25_retriever(chunked_documents)
            hybrid_retriever = EnsembleRetriever(
                retrievers=[semantic_retriever, bm25_retriever],
                weights=[0.8, 0.2]
            )
            st.session_state["retriever_for"] = pdf_path
            st.session_state["semantic_retriever"] = semantic_retriever
            st.session_state["bm25_retriever"] = bm25_retriever
            st.session_state["hybrid_retriever"] = hybrid_retriever
    else:
        semantic_retriever = st.session_state.get("semantic_retriever")
        bm25_retriever = st.session_state.get("bm25_retriever")
        hybrid_retriever = st.session_state.get("hybrid_retriever")
    
    st.subheader("Ask a question about the PDF")
    question = st.chat_input("Type your question here")

    if question:
        # Per-query override for web inclusion
        include_web_for_query = False
        if DDGS_AVAILABLE:
            include_web_for_query = st.checkbox("Include web snippets for this query", value=include_web_by_default)
        else:
            include_web_for_query = False

        st.chat_message("user").write(question)
        
        # Get relevant documents
        related_documents = hybrid_retriever.invoke(question)

        # Optional LLM-based reranking
        if enable_rerank and related_documents:
            # cache key per query+pdf
            cache_key = f"rerank:{pdf_path}:{hash(question)}"
            if cache_key in st.session_state:
                related_documents = st.session_state[cache_key]
            else:
                try:
                    related_documents = llm_rerank(question, related_documents)
                    st.session_state[cache_key] = related_documents
                except Exception:
                    # fallback to original order
                    pass
        
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
            # If web snippets are requested, fetch and append to context
            web_snippets = ""
            if include_web_for_query:
                web_snippets = get_web_content(question)
                # Show a preview panel of web snippets
                try:
                    st.subheader("Web Results Preview")
                    st.text(web_snippets if len(web_snippets) < 5000 else web_snippets[:5000] + "... (truncated)")
                except Exception:
                    st.info("Web results retrieved (preview unavailable)")

            # Pass web snippets into answer generator by stitching them as an extra Document
            if web_snippets:
                from langchain_core.documents import Document
                web_doc = Document(page_content=f"[WEB_CONTEXT]\n{web_snippets}")
                docs_for_answer = related_documents + [web_doc]
            else:
                docs_for_answer = related_documents

            # Append an image metadata document so the model can reference images if needed
            try:
                from langchain_core.documents import Document
                if images_info:
                    img_lines = []
                    for img in images_info:
                        tag = "[screenshot]" if img.get('screenshot') else "[embedded]"
                        img_lines.append(f"{tag} Page {img['page']}: {img['filename']}")
                    images_doc = Document(page_content="[IMAGE_LIST]\n" + "\n".join(img_lines))
                    docs_for_answer = docs_for_answer + [images_doc]
            except Exception:
                # If Document import fails, skip adding image doc
                pass

            answer = answer_question(question, docs_for_answer, show_reasoning=show_reasoning)
            message_placeholder = st.chat_message("assistant")
            message_placeholder.write(answer)
            
            # Display relevant images if any (prefer cropped screenshots covering retrieved chunks)
            display_relevant_images(images_info, answer, question, pdf_path=pdf_path, related_documents=related_documents)
            
            # Add feedback buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Good Answer", key=f"good_answer_{int(time.time()*1000)}"):
                    st.success("Thank you for your feedback!")
            with col2:
                if st.button("👎 Bad Answer", key=f"bad_answer_{int(time.time()*1000)}"):
                    st.error("We'll try to improve!")
                    
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            st.info("Try rephrasing your question or uploading a different PDF.")
        

