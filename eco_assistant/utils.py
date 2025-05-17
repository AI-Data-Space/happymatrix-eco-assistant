"""
Utility functions for the ECO Assistant.

Helper functions for document loading, vector store creation,
and other common operations needed by the main assistant.

Author: Olga Seymour
Date: May 2025
Github: https://github.com/AI-Data-Space/happymatrix-eco-assistant  
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
    """Load and tag ECO documents from a folder with their ECO numbers.

    Extracts ECO numbers from filenames (assuming ECO-XXXXXX format).

    Args:
        folder_path: Path to the folder containing ECO text files

    Returns:
        List of Document objects with ECO tags and metadata
    """
    # Basic validation
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")

    # Find all txt files - need to handle empty directories gracefully
    file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
    if not file_paths:
        print(f"Warning: No .txt files found in {folder_path}")
        print(f"Files in directory: {os.listdir(folder_path)}")

    # Process each file
    documents = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            filename = os.path.basename(path)
            # Get ECO number from filename or use placeholder
            eco_number = filename.split(".")[0] if "ECO-" in filename else "Unknown-ECO"
            tagged_content = f"ECO Number: {eco_number}\n\n{content}"
            documents.append(Document(page_content=tagged_content, metadata={"source": path}))

    print(f"\nAll {len(documents)} ECOs loaded successfully.")
    return documents


def create_vector_db(documents, embedding_model, persist_dir):
    """Create a vector DB from documents using the provided embedding model.

    I experimented with different chunk sizes and found 750/250 worked best
    for our ECO document format. This can be adjusted based on
    specific documents.

    Args:
        documents: List of Document objects to process
        embedding_model: Model to use for embedding generation
        persist_dir: Directory to store the ChromaDB database

    Returns:
        A tuple of (ChromaDB instance, list of split document chunks)
    """
    # First split the docs into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,  # This worked better than 500 or 1000 in testing
        chunk_overlap=250  # ~1/3 overlap seems optimal
    )

    # Split documents
    split_docs = text_splitter.split_documents(documents)

    # Create vector DB
    print(f"Creating {len(split_docs)} document chunks...")
    db = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    return db, split_docs


def call_with_retry(func, max_retries=3, base_delay=5):
    """Helper to handle API rate limits with exponential backoff.

    Args:
        func: Function to call
        max_retries: Max number of retry attempts (default: 3)
        base_delay: Initial delay in seconds, doubles each retry

    Returns:
        Whatever the function returns

    Raises:
        Passes through any non-rate-limit exceptions
        Re-raises rate limit exceptions after max retries
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            # Look for rate limit errors - usually "ResourceExhausted" or 429 status
            if "ResourceExhausted" in str(e) or "429" in str(e):
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                if attempt == max_retries - 1:
                    print("Max retries reached. Please try again later.")
                    raise
            else:
                # Just re-raise other errors
                raise


def validate_eco_number(eco_number):
    """Check if an ECO number has the correct format (ECO-XXXXXX).

    Args:
        eco_number: String to validate

    Returns:
        The validated ECO number (unchanged)

    Raises:
        ValueError: If format doesn't match the expected pattern
    """
    pattern = r"^ECO-\d{6}$"
    if not re.match(pattern, eco_number):
        raise ValueError(f"Invalid ECO number format: {eco_number}. Expected format: ECO-XXXXXX")
    return eco_number


def cleanup_vector_store(persist_dir="eco_chroma_db"):
    """Delete the ChromaDB vector store directory if it exists.

    This is helpful when you need to rebuild the database from scratch.

    Args:
        persist_dir: Path to the ChromaDB directory

    Returns:
        True if deleted, False if not found
    """
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"Deleted existing vector store at: {persist_dir}")
        return True
    else:
        print(f"No vector store found at: {persist_dir}")
        return False
