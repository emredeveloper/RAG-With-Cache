from setuptools import find_packages, setup

setup(
    name="rag_with_cache",
    version="1.1.0",
    packages=find_packages(include=["rag", "rag.*"]),
    install_requires=[
        "faiss-cpu",
        "sentence-transformers",
        "transformers",
        "torch",
        "numpy",
        "pypdf",
        "pytest",
        "inquirer",
    ],
)
