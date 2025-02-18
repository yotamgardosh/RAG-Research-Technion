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

## Results & Benchmarking

### Summary of Results

The evaluation of models with and without RAG augmentation provides insights into how retrieval-based strategies impact model performance. Here's a summary of some initial results:

| Model          | Accuracy (No RAG) | Accuracy (With RAG) |
|----------------|-------------------|---------------------|
| Llama 2 7B     | 78.12%            | 81.25%              |
| Mistral 7B     | 79.38%            | 87.5%               |
| Orca 2 7B      | 79.38             | 83.12%              |
| Gemma 2 2B     | 63.75%            | 75.0%               |


## Conclusion

This project demonstrates the potential of integrating retrieval systems with large language models for more accurate, informed responses in the field of System Engineering. This repository provides a foundation for future research and practical applications in domains that require high-quality question answering and reasoning.

## Contact

For any questions or collaboration inquiries, feel free to contact:

- **[Asaf Shiloah, Yotam Gardosh]**
- **Email: asafshiloah@gmail.com, yotam181@gmail.com**
- **GitHub: [AsafShiloah](https://github.com/asafshiloah), [yotamgardosh](https://github.com/yotamgardosh)**


