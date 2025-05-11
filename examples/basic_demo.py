"""
Basic demonstration of the HappyMatrix ECO Assistant.

This script shows how to initialize the ECO Assistant,
load documents, and perform basic Q&A operations.

Usage:
    python basic_demo.py

Author: Olga Seymour
Date: May 2025
Github: https://github.com/data-ai-studio/happymatrix-eco-assistant
"""

import os
import json
from dotenv import load_dotenv
from eco_assistant import ECOAssistant, CONFIG
from eco_assistant.utils import cleanup_vector_store


def run_demo():
    """Run a basic demonstration of the ECO Assistant capabilities."""
    
    # Get absolute path to SYNT_DOCS relative to the script location
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    synt_docs_path = os.path.join(project_root, "SYNT_DOCS")
    
    # Step 1: Clean up any existing vector store
    cleanup_vector_store()
    
    # Step 2: Load API key from .env file
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("API key not found. Make sure it's set in the .env file.")
    print("API key loaded successfully.")
    
    # Step 3: Initialize the ECO Assistant
    assistant = ECOAssistant(api_key=api_key, config=CONFIG)
    
    try:
        # Step 4: Load ECO documents
        print("Loading ECO documents...")
        documents = assistant.load_documents(synt_docs_path)
        
        # Step 5: Create vector store
        print("Creating vector store...")
        assistant.create_vector_store()
        
        # Step 6: Build QA chain with examples
        print("Building QA chain...")
        examples = """
        Q: For ECO-100001, extract the following fields: Title, Description of Change, 
           Reason for Change, Affected Parts, and Effective Date.
        A:
        Title: Enclosure Update – Add Ventilation Slots
        Description of Change: Ventilation slots were added to the top shell to improve thermal performance.
        Reason for Change: Improve thermal performance.
        Affected Parts: PRT-000210
        Effective Date: 2025-05-01
        
        Q: For ECO-100002, extract the following fields: Title, Description of Change, 
           Reason for Change, Affected Parts, and Effective Date.
        A:
        Title: Battery Type Replacement – Lithium Polymer to Solid-State
        Description of Change: Replaced lithium-polymer battery with solid-state battery in the MatrixSync X100.
        Reason for Change: Improve battery safety, increase product lifespan, and align with new supplier standards.
        Affected Parts: BAT-000011, BAT-000014, BOM-000122
        Effective Date: 2025-05-05
        """
        
        assistant.build_qa_chain(examples)
        
        # Step 7: Run a test query
        print("\n=== Running Test Query ===")
        test_question = "For ECO-100002, what changed and why?"
        
        print(f"Question: {test_question}")
        result = assistant.query(test_question)
        print(f"Answer: {result['result']}")
        
        # Step 8: Get structured output
        print("\n=== Structured JSON Output ===")
        structured_result = assistant.get_structured_output(
            "Extract details from ECO-100001"
        )
        print(json.dumps(structured_result, indent=2))
        
        print("\nDemo completed successfully!")
    
    except Exception as e:
        print(f"Error in demo: {e}")
    
    finally:
        # Always clean up resources
        assistant.cleanup()


if __name__ == "__main__":
    run_demo() 
