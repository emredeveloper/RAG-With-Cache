from setuptools import setup, find_packages

setup(
    name="rag_deepseek",
    version="1.0.0",  # Release sürümüyle eşleşmeli
    packages=find_packages(),
    install_requires=[
        "datasets",
        "faiss-cpu",
        "sentence-transformers",
        "transformers",
        "torch",
        "numpy",
        "pandas",
        "rouge-score",
        "mauve-text",
        "tqdm",
        "pytest",
    ],
)