import os
import requests
import logging
import shutil
from bs4 import BeautifulSoup
import pdfplumber
import markdown
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from chonkie import TokenChunker, TableChunker, SemanticChunker
from ddgs import DDGS
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from langchain_core.prompts import PromptTemplate
from rich.prompt import Prompt as RichPrompt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agentic_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize console for Rich UI
console = Console()

from langchain_community.document_loaders import PyMuPDFLoader

# Initialize Ollama LLM
llm = ChatOllama(
    model="granite4:tiny-h",  # Assuming this is the model used in the project
    temperature=0,
)

def check_local_knowledge(query, context):
    """Router function to determine if we can answer from local knowledge"""
    prompt = '''Role: Question-Answering Assistant
Task: Determine whether the system can answer the user's question based on the provided text.
Instructions:
    - Analyze the text and identify if it contains the necessary information to answer the user's question.
    - Provide a clear and concise response indicating whether the system can answer the question or not.
    - Your response should include only a single word. Nothing else, no other text, information, header/footer.
Output Format:
    - Answer: Yes/No

Study the below examples and based on that, respond to the last question.
Examples:
    Input:
        Text: The capital of France is Paris.
        User Question: What is the capital of France?
    Expected Output:
        Answer: Yes
    Input:
        Text: The population of the United States is over 330 million.
        User Question: What is the population of China?
    Expected Output:
        Answer: No
    Input:
        User Question: {query}
        Text: {text}
'''
    formatted_prompt = prompt.format(text=context, query=query)
    response = llm.invoke(formatted_prompt)
    return response.content.strip().lower() == "yes"

def get_web_content(query: str) -> str:
    """Get web content using DuckDuckGo search with multiple backends."""
    try:
        # Try multiple backends for better results
        backends = ["text", "news"]
        all_results = []

        for backend in backends:
            try:
                with DDGS() as ddgs:
                    if backend == "news":
                        results = ddgs.news(query, max_results=3)
                    else:
                        results = ddgs.text(query, max_results=3)
                    if results:
                        all_results.extend(results)
            except Exception as e:
                console.print(f"[yellow]Warning: {backend} backend failed: {e}[/yellow]")
                continue

        if not all_results:
            return "No web content found."

        # Extract and combine content
        web_content = []
        for result in all_results[:5]:  # Limit to top 5
            title = result.get('title', 'No title')
            snippet = result.get('body', '')  # DDGS uses 'body' instead of 'snippet'
            link = result.get('url', '')  # DDGS uses 'url' instead of 'link'

            # Try to get more content from the link if snippet is empty
            if not snippet and link:
                try:
                    response = requests.get(link, timeout=5)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        # Extract main content
                        content = soup.get_text()
                        snippet = content[:500] + "..." if len(content) > 500 else content
                except Exception as e:
                    console.print(f"[yellow]Could not fetch content from {link}: {e}[/yellow]")

            if snippet:
                # Add source attribution
                source_url = link if link else "Unknown"
                web_content.append(f"[WEB: {source_url}] {title}\nContent: {snippet}")

        return "\n\n".join(web_content) if web_content else "No web content found."

    except Exception as e:
        console.print(f"[red]Web search error: {e}[/red]")
        return "Web search failed."

def extract_tables_from_pdf(pdf_path):
    """Extract tables from PDF and convert to markdown format"""
    tables_markdown = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if table:
                        # Convert table to markdown
                        markdown_table = table_to_markdown(table)
                        tables_markdown.append(f"**Table {table_idx + 1} (Page {page_num + 1}):**\n{markdown_table}\n")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not extract tables from {pdf_path}: {e}[/yellow]")
    
    return "\n".join(tables_markdown) if tables_markdown else ""

def extract_figures_from_pdf(pdf_path):
    """Extract figure captions and descriptions from PDF"""
    figures_info = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    # Look for figure captions (common patterns)
                    lines = text.split('\n')
                    for i, line in enumerate(lines):
                        line_lower = line.lower().strip()
                        if any(keyword in line_lower for keyword in ['figure', 'fig.', 'şekil', 'grafik', 'chart']):
                            # Extract figure description
                            figure_desc = line.strip()
                            # Try to get more context from surrounding lines
                            if i + 1 < len(lines):
                                next_line = lines[i + 1].strip()
                                if next_line and len(next_line) > 10:
                                    figure_desc += " " + next_line
                            figures_info.append(f"**Figure (Page {page_num + 1}):** {figure_desc}")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not extract figures from {pdf_path}: {e}[/yellow]")
    
    return "\n".join(figures_info) if figures_info else ""

def extract_images_from_pdf(pdf_path, output_dir=None):
    """Extract embedded images from a PDF using PyMuPDF (fitz).

    Saves images into output_dir (defaults to <pdf_dir>/extracted_images) and
    returns a list of saved file paths.
    """
    try:
        import fitz
    except Exception as e:
        console.print(f"[yellow]PyMuPDF (fitz) not available: {e}[/yellow]")
        return []

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(pdf_path) or '.', 'extracted_images')
    os.makedirs(output_dir, exist_ok=True)

    saved = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get('image')
                ext = base_image.get('ext', 'png')
                filename = f"page_{page_num+1}_img_{img_index+1}.{ext}"
                out_path = os.path.join(output_dir, filename)
                try:
                    with open(out_path, 'wb') as f:
                        f.write(image_bytes)
                    saved.append(out_path)
                except Exception as e:
                    console.print(f"[yellow]Failed to save image {filename}: {e}[/yellow]")
        doc.close()
    except Exception as e:
        console.print(f"[yellow]Error extracting images from {pdf_path}: {e}[/yellow]")

    if saved:
        console.print(f"[green]Extracted {len(saved)} images from {os.path.basename(pdf_path)} to {output_dir}[/green]")
    return saved

def table_to_markdown(table):
    """Convert a 2D table list to markdown format"""
    if not table or not table[0]:
        return ""
    
    # Create header
    header = "| " + " | ".join(str(cell) if cell else "" for cell in table[0]) + " |"
    separator = "|" + "|".join("---" for _ in table[0]) + "|"
    
    # Create rows
    rows = []
    for row in table[1:]:
        if row:  # Skip empty rows
            row_str = "| " + " | ".join(str(cell) if cell else "" for cell in row) + " |"
            rows.append(row_str)
    
    return "\n".join([header, separator] + rows)

def setup_vector_db(pdf_paths, persist_directory="./chroma_db"):
    """Setup vector database from PDF files with enhanced table and figure extraction"""
    import os
    
    # Check if persistent DB exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        console.print(f"[green]Loading existing vector database from {persist_directory}[/green]")
        try:
            embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
            vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            return vector_db
        except Exception as e:
            console.print(f"[yellow]Could not load existing DB: {e}, creating new one[/yellow]")
    
    all_documents = []
    all_tables = []
    all_figures = []
    
    for pdf_path in pdf_paths:
        # Extract text content
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)
        # Extract embedded images into <pdf_dir>/extracted_images
        try:
            extract_images_from_pdf(pdf_path)
        except Exception:
            pass
        
        # Extract tables
        tables = extract_tables_from_pdf(pdf_path)
        if tables:
            all_tables.append(f"PDF: {pdf_path}\n{tables}")
        
        # Extract figures
        figures = extract_figures_from_pdf(pdf_path)
        if figures:
            all_figures.append(f"PDF: {pdf_path}\n{figures}")
    
    # Combine text and table content for chunking
    combined_content = []
    for doc in all_documents:
        content = doc.page_content
        combined_content.append(content)
    
    # Add extracted tables as separate documents
    for table_content in all_tables:
        combined_content.append(table_content)
    
    # Add extracted figures as separate documents
    for figure_content in all_figures:
        combined_content.append(figure_content)

    # Use Chonkie for chunking - multiple strategies
    token_chunker = TokenChunker()
    table_chunker = TableChunker()
    semantic_chunker = SemanticChunker()

    chunks = []
    for text in combined_content:
        # Try table chunking first
        table_chunks = table_chunker.chunk(text)
        if table_chunks:
            chunks.extend([chunk.text for chunk in table_chunks])
        else:
            # Try semantic chunking for better coherence
            semantic_chunks = semantic_chunker.chunk(text)
            if semantic_chunks:
                chunks.extend([chunk.text for chunk in semantic_chunks])
            else:
                # Fall back to token chunking
                doc_chunks = token_chunker.chunk(text)
                chunks.extend([chunk.text for chunk in doc_chunks])

    # Create persistent vector database
    embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
    vector_db = Chroma.from_texts(
        texts=chunks, 
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Persist the database
    vector_db.persist()
    console.print(f"[green]Vector database persisted to {persist_directory}[/green]")

    return vector_db

def get_local_content(vector_db, query):
    """Get content from vector database with intelligent filtering"""
    # Get more results initially for better filtering
    docs = vector_db.similarity_search(query, k=15)
    
    # Filter and rank results based on relevance and content type
    filtered_docs = []
    table_docs = []
    figure_docs = []
    text_docs = []
    
    for doc in docs:
        content = doc.page_content
        if "Table" in content and "|" in content:
            table_docs.append(doc)
        elif "Figure" in content or "Fig." in content:
            figure_docs.append(doc)
        else:
            text_docs.append(doc)
    
    # Prioritize based on query type
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in ['table', 'tablo', 'data', 'veriler']):
        # Query is about tables, prioritize table content
        filtered_docs = (table_docs + text_docs + figure_docs)[:10]
    elif any(keyword in query_lower for keyword in ['figure', 'fig', 'görsel', 'grafik', 'chart']):
        # Query is about figures, prioritize figure content
        filtered_docs = (figure_docs + text_docs + table_docs)[:10]
    else:
        # General query, balance all types
        filtered_docs = (text_docs + table_docs + figure_docs)[:10]
    
    # Add source attribution to each document
    enhanced_content = []
    for doc in filtered_docs:
        content = doc.page_content
        # Try to extract page info from metadata if available
        page_info = ""
        if hasattr(doc, 'metadata') and doc.metadata:
            page = doc.metadata.get('page', '')
            if page:
                page_info = f" [Page {page}]"
        
        enhanced_content.append(f"[LOCAL{page_info}] {content}")
    
    return " ".join(enhanced_content)

def parse_query_intent(query, default="both"):
    """Parse user query to determine search intent

    Returns one of: 'local_only', 'web_only', 'both'. If no explicit
    preference is found in the query, returns the provided default.
    """
    query_lower = query.lower()

    # Check for explicit source preferences
    if any(keyword in query_lower for keyword in ['sadece pdf', 'only pdf', 'just pdf', 'pdf only', 'local only', 'sadece yerel']):
        return "local_only"
    elif any(keyword in query_lower for keyword in ['sadece web', 'only web', 'just web', 'web only', 'internet only', 'sadece internet']):
        return "web_only"
    elif any(keyword in query_lower for keyword in ['hem pdf hem web', 'both', 'pdf and web', 'local and web']):
        return "both"

    return default
def detect_query_task(query):
    """Detect the type of task from the query"""
    query_lower = query.lower()
    
    if any(keyword in query_lower for keyword in ['özet', 'summary', 'summarize', 'özetle']):
        return "summary"
    elif any(keyword in query_lower for keyword in ['tablo', 'table', 'data', 'veriler', 'show table']):
        return "table_extraction"
    elif any(keyword in query_lower for keyword in ['analiz', 'analysis', 'analyze', 'incele']):
        return "analysis"
    elif any(keyword in query_lower for keyword in ['karşılaştır', 'compare', 'comparison']):
        return "comparison"
    elif any(keyword in query_lower for keyword in ['açıkla', 'explain', 'anlat']):
        return "explanation"
    else:
        return "general_qa"

def export_to_html(markdown_content, query, pdf_paths=None):
    """Export markdown content to HTML file and include tables/figures/images from PDFs.

    pdf_paths: optional list of pdf paths to scan for extracted images and extracted tables/figures.
    """
    try:
        # Convert markdown to HTML
        html_content = markdown.markdown(markdown_content, extensions=['tables', 'fenced_code'])

        # Collect additional assets (tables, figures, images)
        assets_html = ""
        assets_dir = None
        if pdf_paths:
            assets_dir = "extracted_assets_" + datetime.now().strftime('%Y%m%d_%H%M%S')
            os.makedirs(assets_dir, exist_ok=True)

            for pdf_path in pdf_paths:
                # Tables and figure captions
                try:
                    tables_md = extract_tables_from_pdf(pdf_path)
                    figures_md = extract_figures_from_pdf(pdf_path)
                    if tables_md:
                        assets_html += f"<h2>Tables extracted from {os.path.basename(pdf_path)}</h2>\n"
                        assets_html += markdown.markdown(tables_md, extensions=['tables']) + "\n"
                    if figures_md:
                        assets_html += f"<h2>Figures extracted from {os.path.basename(pdf_path)}</h2>\n"
                        assets_html += markdown.markdown(figures_md) + "\n"
                except Exception:
                    # Continue even if extraction fails for one PDF
                    pass

                # Copy any extracted images (common folders)
                possible_dirs = [
                    os.path.join(os.path.dirname(pdf_path), "extracted_images"),
                    os.path.join("hybrid-retrieval-rag", "pdfs", "extracted_images"),
                    os.path.join("hybrid-retrieval-rag", "pdfs")
                ]

                for d in possible_dirs:
                    if os.path.isdir(d):
                        for fname in os.listdir(d):
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                                src = os.path.join(d, fname)
                                dst_name = f"{os.path.basename(pdf_path)}_{fname}"
                                dst = os.path.join(assets_dir, dst_name)
                                try:
                                    shutil.copyfile(src, dst)
                                    assets_html += f"<div><img src=\"{os.path.join(assets_dir, dst_name)}\" style=\"max-width:100%;height:auto\"><p>{fname} (from {os.path.basename(pdf_path)})</p></div>\n"
                                except Exception:
                                    continue

        # Add basic HTML styling
        full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agentic RAG Results - {query[:50]}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .query-info {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="query-info">
            <h1>🤖 Agentic RAG Results</h1>
            <p><strong>Query:</strong> {query}</p>
            <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    {html_content}
    {assets_html}
    </div>
</body>
</html>
"""
        
        # Save to file
        filename = f"rag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        console.print(f"[green]Results exported to {filename}[/green]")
        logger.info(f"HTML results exported to {filename}")
        return filename
        
    except Exception as e:
        console.print(f"[red]Error exporting to HTML: {e}[/red]")
        logger.error(f"Error exporting to HTML: {e}")
        return None

def get_dynamic_prompt(task_type):
    """Get appropriate prompt based on detected task"""
    base_template = """
You are a helpful assistant. Answer the user's query using the provided contexts.

LOCAL CONTEXT (from PDF documents):
{local_context}

WEB CONTEXT (from internet search):
{web_context}

QUERY: {query}

INSTRUCTIONS:
- Provide a comprehensive answer structured in clear sections
- Start with a direct answer to the query
- Include relevant information from both web and local sources
- Use source tags like [LOCAL: page X] and [WEB: url] for attribution
- Be detailed but organized, use headers and bullet points
"""
    
    if task_type == "summary":
        return base_template + """
- Focus on summarizing key points and main ideas
- Keep the summary concise but comprehensive
- Highlight the most important information

STRUCTURE YOUR ANSWER LIKE THIS:
# Summary
[Concise summary of the main points]

# Key Points
[Bullet points of important information]

# Sources
[List of sources used]
"""
    
    elif task_type == "table_extraction":
        return base_template + """
- Focus on extracting and displaying tables from the content
- Convert any tabular data to markdown table format
- Include table titles and descriptions

STRUCTURE YOUR ANSWER LIKE THIS:
# Tables Found
[Display all relevant tables in markdown format]

# Table Analysis
[Analysis of the table data]

# Sources
[Source attribution for tables]
"""
    
    elif task_type == "analysis":
        return base_template + """
- Provide in-depth analysis of the topic
- Break down complex concepts
- Use logical reasoning and evidence from sources

STRUCTURE YOUR ANSWER LIKE THIS:
# Analysis
[Detailed analysis with reasoning]

# Key Insights
[Important findings and insights]

# Conclusion
[Final conclusions based on analysis]
"""
    
    elif task_type == "comparison":
        return base_template + """
- Compare different aspects, methods, or concepts
- Use clear comparison criteria
- Highlight similarities and differences

STRUCTURE YOUR ANSWER LIKE THIS:
# Comparison Overview
[Brief overview of what is being compared]

# Detailed Comparison
[Point-by-point comparison]

# Recommendation
[If applicable, recommendations based on comparison]
"""
    
    else:  # general_qa or explanation
        return base_template + """
STRUCTURE YOUR ANSWER LIKE THIS:
# Direct Answer
[Brief direct answer to the query]

# Detailed Explanation
[Comprehensive explanation with evidence]

# Additional Information
[Any relevant additional details]

# Sources
[Complete source attribution]
"""

def generate_final_answer(local_context, web_context, query):
    """Generate final answer using LLM with dynamic prompting"""
    task_type = detect_query_task(query)
    template = get_dynamic_prompt(task_type)
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["local_context", "web_context", "query"]
    )

    chain = prompt | llm
    response = chain.invoke({
        "local_context": local_context,
        "web_context": web_context,
        "query": query
    })

    return response.content

def process_query(query, vector_db, local_context, console, default_include_web: bool = False):
    """Main function to process user query with enhanced error handling"""
    try:
        logger.info(f"Processing query: {query}")
        console.print(f"[bold blue]Processing query:[/bold blue] {query}")

        # Parse user intent for search scope
        default_scope = "both" if default_include_web else "local_only"
        search_intent = parse_query_intent(query, default=default_scope)
        logger.info(f"Search intent: {search_intent}")

        # Step 1: Get local context based on intent
        local_ctx = ""
        if search_intent in ["local_only", "both"]:
            try:
                local_ctx = get_local_content(vector_db, query)
                console.print("[yellow]Retrieved context from local documents[/yellow]")
                logger.info(f"Retrieved {len(local_ctx.split())} words from local documents")
            except Exception as e:
                logger.error(f"Error retrieving local context: {e}")
                local_ctx = "Error retrieving local context"
                console.print(f"[red]Error retrieving local context: {e}[/red]")

        # Step 2: Get web context based on intent
        web_ctx = ""
        if search_intent in ["web_only", "both"]:
            try:
                web_ctx = get_web_content(query)
                console.print("[red]Retrieved context from web search[/red]")
                # Display a short preview of web results to the user
                try:
                    preview = web_ctx if len(web_ctx) < 2000 else web_ctx[:2000] + "... (truncated)"
                    console.print(Panel(preview, title="Web Results Preview", border_style="magenta"))
                except Exception:
                    console.print("[yellow]Web results retrieved (unable to display preview).[/yellow]")
                logger.info(f"Retrieved web context with {len(web_ctx.split())} words")
            except Exception as e:
                logger.error(f"Error retrieving web context: {e}")
                web_ctx = "Error retrieving web context"
                console.print(f"[red]Error retrieving web context: {e}[/red]")

        # Step 3: Combine contexts
        combined_context = ""
        if local_ctx:
            combined_context += f"Local knowledge: {local_ctx}\n\n"
        if web_ctx:
            combined_context += f"Web knowledge: {web_ctx}"

        # Step 4: Generate final answer
        try:
            answer = generate_final_answer(local_ctx, web_ctx, query)
            logger.info("Successfully generated final answer")
            return answer
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            console.print(f"[red]Error generating answer: {e}[/red]")
            return f"Error generating answer: {str(e)}"
            
    except Exception as e:
        logger.error(f"Unexpected error in process_query: {e}")
        console.print(f"[red]Unexpected error: {e}[/red]")
        return f"Unexpected error: {str(e)}"

def main():
    """Main application with comprehensive error handling"""
    try:
        console = Console()
        logger.info("Starting Agentic RAG system")

        # Header
        console.print(Panel.fit(
            "[bold magenta]🤖 Agentic RAG System with Ollama[/bold magenta]\n"
            "[dim]Using Chonkie chunking, ChromaDB, and DuckDuckGo search[/dim]",
            title="Welcome",
            border_style="blue"
        ))

        # Use only the specified PDF
        pdf_paths = ["2405.12981v1.pdf"]
        logger.info(f"Using PDF paths: {pdf_paths}")

        # Initialize vector database
        try:
            console.print("[bold green]Setting up vector database...[/bold green]")
            vector_db = setup_vector_db(pdf_paths)
            logger.info("Vector database setup completed")
        except Exception as e:
            logger.error(f"Failed to setup vector database: {e}")
            console.print(f"[red]Failed to setup vector database: {e}[/red]")
            return

        # Get initial context from PDFs for routing
        try:
            local_context = get_local_content(vector_db, "artificial intelligence")
            logger.info("Initial context loaded")
        except Exception as e:
            logger.error(f"Failed to load initial context: {e}")
            local_context = ""
            console.print(f"[yellow]Warning: Could not load initial context: {e}[/yellow]")

        # Ask user whether to include web search results by default
        include_web_default = False
        try:
            include_web_choice = RichPrompt.ask("Include web search results by default? (y/n)", default="n")
            include_web_default = include_web_choice.lower() in ['y', 'yes']
        except Exception:
            # If prompt fails (non-interactive), default to False
            include_web_default = False

        console.print("[green]✅ System ready! Ask me anything about AI or your documents.[/green]")
        logger.info("System ready for queries")

        while True:
            try:
                # Get user query
                query = RichPrompt.ask("\n[bold cyan]Your question[/bold cyan]")
                logger.info(f"User query: {query}")

                if query.lower() in ['exit', 'quit', 'q']:
                    console.print("[yellow]Goodbye! 👋[/yellow]")
                    logger.info("User exited the system")
                    break

                result = process_query(query, vector_db, local_context, console, default_include_web=include_web_default)

                # Display final answer in a panel with markdown support
                markdown_answer = Markdown(result)
                console.print(Panel(
                    markdown_answer,
                    title="[bold green]Final Answer[/bold green]",
                    border_style="green"
                ))

                # Ask user if they want to export to HTML
                export_choice = RichPrompt.ask("\n[bold yellow]Export to HTML?[/bold yellow] (y/n)", default="n")
                if export_choice.lower() in ['y', 'yes']:
                    export_filename = export_to_html(result, query, pdf_paths=pdf_paths)
                    if export_filename:
                        console.print(f"[green]✅ Results saved as {export_filename}[/green]")

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted by user. Goodbye! 👋[/yellow]")
                logger.info("System interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                console.print(f"[red]Error in main loop: {str(e)}[/red]")
                continue
                
    except Exception as e:
        logger.critical(f"Critical error in main: {e}")
        console.print(f"[red]Critical error: {str(e)}[/red]")
    finally:
        logger.info("Agentic RAG system shutting down")

if __name__ == "__main__":
    main()