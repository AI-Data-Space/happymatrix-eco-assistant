"""
Configuration settings for the ECO Assistant.

This module defines default settings for the ECO Assistant,
including model parameters, vector database settings, and
retrieval options.

Author: Olga Seymour
Date: May 2025
Github: https://github.com/AI-Data-Space/happymatrix-eco-assistant   
"""

# Default configuration for ECO Assistant
CONFIG = {
    "chunk_size": 750,                          # Size of text chunks for embedding 
    "chunk_overlap": 250,                       # Overlap prevents context loss between chunks    
    "retriever_k": 8,                           # Number of chunks to retrieve in queries 
    "persist_dir": "eco_chroma_db",             # Vector DB storage location 
    "model": "models/gemini-1.5-flash",         # LLM model. Using the faster Gemini model for better latency 
    "embedding_model": "models/embedding-001",  # Google's text embedding model  
}
