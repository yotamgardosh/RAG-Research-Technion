# RAG-Research-Technion

## 🚀 Overview

**RAG-Research-Technion** is an advanced **Retrieval-Augmented Generation (RAG) system** designed to enhance Large Language Model (LLM) responses by dynamically retrieving relevant context from external knowledge sources. By leveraging **LangChain**, **ChromaDB**, and **vector search**, the system improves **domain-specific question answering and reasoning**.

This project integrates ** open-source Foundation models LLMs** such as:
- **Mistral 7B**
- **Orca 2 7B**
- **Llama 2 13B**
- **Gemma 2 2B**

## 🔧 Features

- **Contextual RAG Implementation**: Retrieves domain-specific knowledge before generating responses.
- **Vector Search with ChromaDB**: Uses **semantic search** to fetch **highly relevant documents**.
- **LangChain Orchestration**: Implements **document retrieval, prompt engineering, and structured LLM execution**.
- **Multi-Model Benchmarking**: Evaluates different **LLMs with and without RAG** for comparative analysis.
- **Chain-of-Thought (CoT) Reasoning**: Guides models to **think in structured steps** for better decision-making.
- **Efficient Document Filtering**: Uses **LLM-as-a-Judge** to reduce noise in retrieved context.

## 📂 Repository Structure
RAG-Research-Technion/ ├── src/ # Source code (retrieval, model inference, evaluation) ├── data/ # Datasets (raw text, embeddings, etc.) ├── models/ # Pre-trained and fine-tuned models ├── results/ # Evaluation outputs and benchmarking results ├── docs/ # Research reports, references, and technical documents ├── notebooks/ # Jupyter notebooks for testing and analysis ├── tests/ # Unit and integration tests ├── README.md # Project documentation (this file) ├── LICENSE # License information └── .gitignore # Files to be ignored in Git

