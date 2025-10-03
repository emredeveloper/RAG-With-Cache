import os
import requests
from bs4 import BeautifulSoup
import pdfplumber
from langchain_community.document_loaders import PyMuPDFLoader
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

# Initialize console for Rich UI
console = Console()

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
                web_content.append(f"Title: {title}\nContent: {snippet}\nSource: {link}")

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

def setup_vector_db(pdf_paths):
    """Setup vector database from PDF files with enhanced table extraction"""
    all_documents = []
    all_tables = []
    
    for pdf_path in pdf_paths:
        # Extract text content
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)
        
        # Extract tables
        tables = extract_tables_from_pdf(pdf_path)
        if tables:
            all_tables.append(f"PDF: {pdf_path}\n{tables}")
    
    # Combine text and table content for chunking
    combined_content = []
    for doc in all_documents:
        content = doc.page_content
        combined_content.append(content)
    
    # Add extracted tables as separate documents
    for table_content in all_tables:
        combined_content.append(table_content)

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

    # Create vector database
    embeddings = OllamaEmbeddings(model="embeddinggemma:latest")  # Using embeddinggemma:latest
    vector_db = Chroma.from_texts(chunks, embeddings)

    return vector_db

def get_local_content(vector_db, query):
    """Get content from vector database"""
    docs = vector_db.similarity_search(query, k=10)  # Increased from 5 to 10 for more comprehensive results
    return " ".join([doc.page_content for doc in docs])

def generate_final_answer(local_context, web_context, query):
    """Generate final answer using LLM with structured prompt"""
    template = """
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
- Use [LOCAL] tags for PDF-sourced information
- Use [WEB] tags for web-sourced information
- If tables exist in local context, display them as properly formatted markdown tables
- Include visual/figure descriptions when available in the PDF
- When the query references specific tables or figures, include them prominently
- End with a conclusion that synthesizes both sources
- Be detailed but organized, use headers and bullet points

STRUCTURE YOUR ANSWER LIKE THIS:
# Direct Answer
[Brief direct answer to the query]

# Web Sources
[Information from web search with [WEB] tags]

# PDF Content & Tables
[Relevant content from PDF with tables in markdown format, [LOCAL] tags]
[Include specific tables/figures referenced in the query]

# Analysis & Conclusion
[Combined analysis synthesizing both sources]

Answer in clean markdown format:"""

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

def process_query(query, vector_db, local_context, console):
    """Main function to process user query"""
    console.print(f"[bold blue]Processing query:[/bold blue] {query}")

    # Step 1: Always get local context
    local_ctx = get_local_content(vector_db, query)
    console.print("[yellow]Retrieved context from local documents[/yellow]")

    # Step 2: Always get web context
    web_ctx = get_web_content(query)
    console.print("[red]Retrieved context from web search[/red]")

    # Step 3: Combine contexts
    combined_context = f"Local knowledge: {local_ctx}\n\nWeb knowledge: {web_ctx}"

    # Step 4: Generate final answer
    answer = generate_final_answer(local_ctx, web_ctx, query)
    return answer

def main():
    console = Console()

    # Header
    console.print(Panel.fit(
        "[bold magenta]🤖 Agentic RAG System with Ollama[/bold magenta]\n"
        "[dim]Using Chonkie chunking, ChromaDB, and DuckDuckGo search[/dim]",
        title="Welcome",
        border_style="blue"
    ))

    # Use only the specified PDF
    pdf_paths = ["2405.12981v1.pdf"]

    # Initialize vector database
    console.print("[bold green]Setting up vector database...[/bold green]")
    vector_db = setup_vector_db(pdf_paths)

    # Get initial context from PDFs for routing
    local_context = get_local_content(vector_db, "artificial intelligence")

    console.print("[green]✅ System ready! Ask me anything about AI or your documents.[/green]")

    while True:
        # Get user query
        query = RichPrompt.ask("\n[bold cyan]Your question[/bold cyan]")

        if query.lower() in ['exit', 'quit', 'q']:
            console.print("[yellow]Goodbye! 👋[/yellow]")
            break

        try:
            result = process_query(query, vector_db, local_context, console)

            # Display final answer in a panel with markdown support
            markdown_answer = Markdown(result)
            console.print(Panel(
                markdown_answer,
                title="[bold green]Final Answer[/bold green]",
                border_style="green"
            ))

        except Exception as e:
            console.print(f"[red]Error processing query: {str(e)}[/red]")

if __name__ == "__main__":
    main()