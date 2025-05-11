"""
Utility functions for the ECO Assistant.

This module provides helper functions for loading documents,
creating vector stores, handling API rate limits, and validating
ECO numbers.

Author: Olga Seymour
Date: May 2025
Github: https://github.com/data-ai-studio/happymatrix-eco-assistant
"""

import os
import glob
import re
import shutil
import time
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


def load_and_tag_documents(folder_path):
    """
    Load ECO documents from a folder and tag them with their ECO number.
    
    Args:
        folder_path (str): Path to folder containing ECO text files
        
    Returns:
        list: List of tagged document texts
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        NotADirectoryError: If path is not a directory
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")
    
    # Find all text files - using glob pattern matching
    file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
    
    # Warn if we don't find any docs - good for debugging 
    if not file_paths:
        print(f"Warning: No .txt files found in {folder_path}")
        print(f"Files in directory: {os.listdir(folder_path)}")
    
    # Load each doc and tag it with its ECO number
    # This helps with retrieval context later
    documents = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            filename = os.path.basename(path)
            
            # Pull ECO number from filename - assumes ECO-XXXXXX format
            eco_number = filename.split(".")[0] if "ECO-" in filename else "Unknown-ECO"
            tagged_content = f"ECO Number: {eco_number}\n\n{content}"
            documents.append(tagged_content)
    
    # Confirmation
    print(f"\nAll {len(documents)} ECOs loaded successfully.")
    return documents


def create_vector_db(documents, embedding_model, persist_dir):
    """
    Create a vector database from documents.
    
    Splits documents into chunks and stores them in ChromaDB with embeddings.
    
    Args:
        documents (list): List of document texts
        embedding_model: Embedding model to use
        persist_dir (str): Directory to store the database
        
    Returns:
        tuple: (ChromaDB instance, list of split documents)
    """
    
    # Split docs into manageable chunks for better embedding
    # Using recursive splitter to respect natural text boundaries 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=250
    )
    docs = [Document(page_content=d) for d in documents]
    split_docs = text_splitter.split_documents(docs)
    
    # Create Chroma vector DB from our doc chunks 
    # This is where the magic happens for semantic search 
    db = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    return db, split_docs


def call_with_retry(func, max_retries=3, base_delay=5):
    """
    Handle API rate limits with exponential backoff.
    
    Retries the function call with increasing delays if rate limited.
    
    Args:
        func (callable): Function to call
        max_retries (int): Maximum number of retry attempts
        base_delay (int): Base delay in seconds (doubles each retry)
        
    Returns:
        The result of the function call
        
    Raises:
        Exception: Raises original exception after max retries
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            # Check specifically for rate limit errors 
            if "ResourceExhausted" in str(e) or "429" in str(e):
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                if attempt == max_retries - 1:
                    print("Max retries reached. Please try again later.")
                    raise
            else:
                # Not a rate limit - just pass through other errors 
                raise


def validate_eco_number(eco_number):
    """
    Validate that the ECO number has the correct format.
    
    Args:
        eco_number (str): ECO number to validate
        
    Returns:
        str: The validated ECO number
        
    Raises:
        ValueError: If ECO number format is invalid
    """
    pattern = r"^ECO-\d{6}$"
    if not re.match(pattern, eco_number):
        raise ValueError(f"Invalid ECO number format: {eco_number}. Expected format: ECO-XXXXXX")
    return eco_number


def cleanup_vector_store(persist_dir="eco_chroma_db"):
    """
    Delete previous ChromaDB vector store.
    
    Args:
        persist_dir (str): Path to the vector store directory
        
    Returns:
        bool: True if deleted, False if not found
    """
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"Deleted existing vector store at: {persist_dir}")
        return True
    else:
        print(f"No vector store found at: {persist_dir}")
        return False
