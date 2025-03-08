import sys
import asyncio
import os
import gc
import tempfile
import uuid
import pandas as pd
import time
import traceback
from gitingest import ingest
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser
import streamlit as st

# Fix for Windows: Set the Proactor event loop policy to support subprocesses
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Session state initialization
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.messages = []

session_id = st.session_state.id

@st.cache_resource
def load_llm():
    """Load the Ollama LLM model with caching."""
    return Ollama(model="qwen2.5:7b", request_timeout=120.0)

@st.cache_resource
def load_embedding_model():
    """Load the Ollama embedding model (nomic-embed-text) with caching."""
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434"  # Default Ollama URL, adjust if necessary
    )
    return embed_model

def reset_chat():
    """Reset the chat history and clear memory."""
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def process_with_gitingets(github_url, retries=3, delay=5):
    """
    Fetch GitHub repository content using the gitingest library.
    
    Args:
        github_url (str): URL of the GitHub repository.
        retries (int): Number of retry attempts for fetching the repository.
        delay (int): Delay between retry attempts in seconds.
    
    Returns:
        tuple: (summary, tree, content) if successful, (None, None, None) otherwise.
    """
    for attempt in range(retries):
        try:
            with st.spinner(f"Fetching repository: {github_url} (Attempt {attempt + 1}/{retries})"):
                summary, tree, content = ingest(github_url)
            st.success("Repository processed successfully!")
            return summary, tree, content
        except Exception as e:
            if attempt < retries - 1:
                # Adjust delay for network-related errors
                error_msg = str(e).lower()
                if "network" in error_msg or "connection" in error_msg:
                    delay = delay * 2  # Double the delay for network issues
                st.warning(f"Retrying after {delay} seconds... Error: {str(e)}")
                time.sleep(delay)
            else:
                st.error(f"Error processing GitHub repo after {retries} attempts:\n\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}")
                return None, None, None

@st.cache_data
def build_index(_docs):
    """Build and cache the VectorStoreIndex from documents.
    
    Args:
        _docs (list): List of documents (not hashed by Streamlit due to leading underscore).
    
    Returns:
        VectorStoreIndex: The built index.
    """
    node_parser = MarkdownNodeParser()
    index = VectorStoreIndex.from_documents(
        documents=_docs,
        transformations=[node_parser],
        show_progress=True
    )
    return index

def setup_query_engine(index, llm):
    """Set up the query engine with a custom prompt template."""
    Settings.llm = llm
    query_engine = index.as_query_engine(streaming=True)
    
    # Customizing prompt template
    qa_prompt_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context, answer concisely. If you don't know, say 'I don't know!'.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    query_engine.update_prompts({"response_synthesizer:text_qa_template": PromptTemplate(qa_prompt_tmpl_str)})
    
    return query_engine

# Main UI layout
st.title("Chat with GitHub using RAG </>")
st.markdown("Enter a GitHub repository URL to chat with its content.")

with st.sidebar:
    st.header("Add your GitHub repository!")
    
    # Input for GitHub repository URL
    github_url = st.text_input("Enter GitHub repository URL", placeholder="GitHub URL")
    
    # Input for GitHub token (with option to use environment variable)
    github_token = st.text_input(
        "Enter GitHub Token (or leave blank to use GITHUB_TOKEN env variable)",
        type="password",
        placeholder="GitHub Token (optional)"
    )
    
    # If a token is provided, set it as the GITHUB_TOKEN environment variable
    if github_token:
        os.environ["GITHUB_TOKEN"] = github_token
    else:
        # Check if GITHUB_TOKEN is already set in the environment
        if not os.getenv("GITHUB_TOKEN"):
            st.warning("No GitHub token provided and GITHUB_TOKEN environment variable not set. This may cause issues with private repositories or rate limits.")
    
    load_repo = st.button("Load Repository")

    if github_url and load_repo:
        if not github_url.startswith("https://github.com/"):
            st.error("Please enter a valid GitHub repository URL (e.g., https://github.com/username/repo).")
            st.stop()

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with st.spinner("Processing your repository..."):
                    repo_name = github_url.split('/')[-1]
                    file_key = f"{session_id}-{repo_name}"
                    
                    if file_key not in st.session_state.file_cache:
                        # Fetch repository content
                        summary, tree, content = process_with_gitingets(github_url)
                        if not content:
                            st.error("Failed to retrieve content from repository.")
                            st.stop()
                        
                        # Save the content to a temporary file
                        content_path = os.path.join(temp_dir, f"{repo_name}_content.md")
                        with open(content_path, "w", encoding="utf-8") as f:
                            f.write(content)
                        
                        # Load the content using LlamaIndex
                        loader = SimpleDirectoryReader(input_dir=temp_dir)
                        docs = loader.load_data()
                        
                        # Load models and build index
                        llm = load_llm()
                        embed_model = load_embedding_model()
                        Settings.embed_model = embed_model
                        
                        # Build and cache the index
                        index = build_index(docs)
                        
                        # Set up query engine
                        query_engine = setup_query_engine(index, llm)
                        
                        st.session_state.file_cache[file_key] = query_engine
                        gc.collect()  # Clean up memory after indexing
                    else:
                        query_engine = st.session_state.file_cache[file_key]
                    
                    st.success("Ready to Chat!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.stop()

# Chat interface
col1, col2 = st.columns([6, 1])
with col1:
    st.header("Chat")
with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            repo_name = github_url.split('/')[-1]
            file_key = f"{session_id}-{repo_name}"
            query_engine = st.session_state.file_cache.get(file_key)
            
            if query_engine is None:
                st.error("Please load a repository first!")
                st.stop()
            
            with st.spinner("Generating response..."):
                response = query_engine.query(prompt)
                
                if hasattr(response, 'response_gen'):
                    for chunk in response.response_gen:
                        if isinstance(chunk, str):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                else:
                    full_response = str(response)
                    message_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            full_response = "Error occurred while processing your request."
            message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})