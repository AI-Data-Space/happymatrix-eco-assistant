"""
HappyMatrix ECO Assistant - Main Demo

This script demonstrates the core functionality of the ECO Assistant,
a GenAI-powered tool for analyzing Engineering Change Orders.

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
    """Run a demonstration of the ECO Assistant's key capabilities."""
    # Clean up any existing vector store
    cleanup_vector_store()
    
    # Load API key from .env file (or set one for this demo)
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("\nNo API key found in .env file.")
        print("To run this demo, you need a Google Gemini API key.")
        print("1. Get an API key from: https://aistudio.google.com/app/apikey")
        print("2. Create a .env file in the project root")
        print("3. Add your key: GOOGLE_API_KEY=your_key_here\n")
        return
    
    print("API key loaded successfully.")
    
    # Create an instance of the ECOAssistant class with API key and config 
    print("\nInitializing ECO Assistant...") 
    assistant = ECOAssistant(api_key=api_key, config=CONFIG)
    
    try:
        # Load ECO documents
        print("\nLoading ECO documents...")
        documents = assistant.load_documents("SYNT_DOCS")
        
        # Create vector store for semantic search 
        print("\nCreating vector database...")
        assistant.create_vector_store()
        
        # Build QA chain with examples
        print("\nBuilding QA chain...")
        examples = """
        Q: For ECO-100001, extract the following fields: Title, Description of Change, Reason for Change, Affected Parts, and Effective Date.
        A:
        Title: Enclosure Update – Add Ventilation Slots
        Description of Change: Ventilation slots were added to the top shell to improve thermal performance.
        Reason for Change: Improve thermal performance.
        Affected Parts: PRT-000210
        Effective Date: 2025-05-01
        """
        assistant.build_qa_chain(examples)
        
        # Demo 1: Simple Q&A
        print("\n\n===== DEMO 1: Simple Q&A =====")
        print("\nQuestion: What change was made in ECO-100002 and why?")
        result = assistant.query("What change was made in ECO-100002 and why?")
        print("\nAnswer:")
        print(result["result"])
        
        # Demo 2: Structured Output (JSON)
        print("\n\n===== DEMO 2: Structured Output (JSON) =====")
        print("\nExtracting structured data from ECO-100001...")
        structured_result = assistant.get_structured_output(
            "Extract details from ECO-100001"
        )
        print("\nStructured JSON result:")
        print(json.dumps(structured_result, indent=2))
        
        # Demo 3: Generate Stakeholder Email
        print("\n\n===== DEMO 3: Stakeholder Email Generation =====")
        print("\nGenerating stakeholder email for ECO-100002...")
        email = assistant.generate_stakeholder_email("ECO-100002")
        print("\nEmail content:")
        print(email)
        
        print("\n\nDemo completed successfully!")
        print("Try more features by exploring the examples/ directory.")
        
    except Exception as e:
        print(f"\nError in demo: {e}")
    
    finally:
        # Always clean up resources
        assistant.cleanup()


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" HappyMatrix ECO Assistant Demo")
    print(" A GenAI-Powered Engineering Change Order Analysis Tool")
    print("="*60)
    run_demo()
