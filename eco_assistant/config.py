"""
Configuration settings for the ECO Assistant.

This module defines default settings for the ECO Assistant,
including model parameters, vector database settings, and
retrieval options.
"""

# Default configuration for ECO Assistant
CONFIG = {
    "chunk_size": 750,                          # Size of text chunks for embedding 
    "chunk_overlap": 250,                       # Overlap between chunks to maintain context  
    "retriever_k": 8,                           # Number of chunks to retrieve in queries 
    "persist_dir": "eco_chroma_db",             # Vector DB storage location 
    "model": "models/gemini-1.5-flash",         # LLM model 
    "embedding_model": "models/embedding-001",  # Embedding model  
}