# genai-eco-assistant
# HappyMatrix ECO Assistant

This project is a Generative AI Assistant designed to work with synthetic Engineering Change Order (ECO) documents. It demonstrates how Large Language Models (LLMs), embeddings, and Retrieval Augmented Generation (RAG) techniques can be applied to product data management workflows.

## Project Overview

- **Objective**: Build an assistant capable of answering questions about ECO processes, extracting structured information, and assisting with compliance checks using synthetic data.
- **Dataset**: 4 synthetic ECO documents representing fictional product changes for a company named HappyMatrix Inc.
- **Approach**:
  - Load and preprocess synthetic ECO documents.
  - Generate document embeddings using `InstructorEmbeddings`.
  - Store embeddings in a Chroma vector database.
  - Implement Retrieval Augmented Generation (RAG) for document Q&A.
  - Use a Gemini-based language model to generate responses.

## Key Components

- **Synthetic Data**: Custom-created ECO documents to simulate real-world engineering change scenarios.
- **Vector Database**: ChromaDB is used to store and retrieve document embeddings.
- **Embeddings Model**: InstructorEmbeddings for better domain-specific understanding.
- **Generative Model**: Gemini Pro (through Vertex AI) for intelligent response generation.
- **Document Q&A**: Retrieval + generation system enabling natural language queries about ECO content.

## Technologies Used

- Python
- LangChain
- ChromaDB
- InstructorEmbedding (text embeddings)
- Vertex AI (Gemini Pro Model)
- Kaggle Notebook environment

## How to Use

1. Load synthetic ECO documents into the assistant.
2. Create embeddings and store them in the vector database.
3. Run Q&A interface to ask questions about ECO content.
4. Retrieve precise answers with citations from source documents.

## Project Structure


## Acknowledgements

This project was inspired by the Google Generative AI Intensive Course and is part of a hands-on capstone project. All ECO documents are fictional and created for demonstration purposes only.

---

