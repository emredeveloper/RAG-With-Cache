import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain import PromptTemplate, LLMChain
from langchain_groq import ChatGroq

# Streamlit UI Configuration
st.set_page_config(page_title="Groq Data Analysis", layout="wide")
st.title("🚀 Enhanced Data Analysis with Groq, LangChain & Visuals")

# API Key Input
st.sidebar.subheader("Groq API Configuration")
groq_api_key = st.sidebar.text_input(
    "Enter Groq API Key:",
    placeholder="Paste your Groq API key here",
    type="password",
    help="You can get your API key from the Groq website."
)

# Ensure API Key is provided
if not groq_api_key:
    st.warning("Please provide your Groq API key in the sidebar to proceed.")
else:
    @st.cache_resource
    def load_groq_llm():
        """Load Groq LLM dynamically with provided API key."""
        llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant", temperature=0.1)
        return llm

    # Data Cleaning
    def clean_data(df):
        """Clean and standardize dataset."""
        df.columns = df.columns.str.replace("[^a-zA-Z0-9]", "", regex=True)
        return df.convert_dtypes()

    # LangChain Analysis Function
    def analyze_with_langchain(df, question):
        """Use LangChain with Groq for data analysis."""
        try:
            llm = load_groq_llm()
            # Prompt template
            template = """
            You are an advanced data analysis assistant. Your task is to analyze datasets and answer questions based on the provided data.
            
            Here is a sample of the dataset (first 10 rows):
            {data}

            Question: {question}

            Instructions:
            - If the question is about filtering (e.g., "cities with a population over 5 million"), list only the matching rows.
            - Provide results as comma-separated values for matching rows.
            - If no data matches, respond with: 'No results found.'
            - Do not include extra details or unrelated information in your response.

            Your response must be clear, concise, and formatted correctly.
            """
            prompt = PromptTemplate(template=template, input_variables=["data", "question"])

            # Prepare data for prompt
            truncated_df = df.head(10)
            data_as_markdown = truncated_df.to_markdown()

            # LangChain LLMChain
            chain = LLMChain(llm=llm, prompt=prompt)
            response = chain.run({"data": data_as_markdown, "question": question})
            return response.strip()
        except Exception as e:
            return f"Error during analysis: {str(e)}"

    # File Upload Section
    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset (CSV/Excel)",
        type=["csv", "xlsx"],
        help="Upload a file to analyze (Max size: 10MB)."
    )

    # Default or Uploaded Data
    if not uploaded_file:
        sample_data = {
            "City": ["New York", "London", "Tokyo", "Paris", "Mumbai"],
            "Population": [8.4, 8.9, 13.9, 2.1, 20.4],
            "Country": ["USA", "UK", "Japan", "France", "India"],
        }
        df = pd.DataFrame(sample_data)
    else:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    df = clean_data(df)

    # UI Components
    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    # Analysis Panel
    question = st.text_input(
        "Ask a Question:",
        placeholder="Which cities have population over 5 million?",
        help="Enter a question related to the dataset."
    )

    if st.button("Run Analysis"):
        with st.spinner("Analyzing the dataset..."):
            try:
                # Run analysis with LangChain
                answer = analyze_with_langchain(df, question)
                st.subheader("Analysis Results")
                st.success(f"Answer: {answer}")

                # Generate a visualization if possible
                if "cities" in question.lower() and "population" in question.lower():
                    # Filter cities with population > 5 million
                    filtered_df = df[df["Population"] > 5]
                    if not filtered_df.empty:
                        st.subheader("Visualization")
                        # Plot data
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(filtered_df["City"], filtered_df["Population"], color="skyblue")
                        ax.set_title("Cities with Population > 5 Million", fontsize=16)
                        ax.set_ylabel("Population (in Millions)", fontsize=12)
                        ax.set_xlabel("City", fontsize=12)
                        st.pyplot(fig)
                    else:
                        st.warning("No cities with a population over 5 million were found.")
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

    # Debug Info
    with st.expander("Debug Info"):
        st.markdown("**Data Statistics:**")
        st.write(df.describe())
        st.markdown("**Column Types:**")
        st.write(df.dtypes)
