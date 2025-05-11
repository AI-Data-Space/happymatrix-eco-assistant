"""
This script runs the key examples from the ECO Assistant notebook and displays the results
to compare with the notebook outputs.

Author: Olga Seymour
Date: May 2025
Github: https://github.com/data-ai-studio/happymatrix-eco-assistant
"""

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv


from eco_assistant import ECOAssistant, CONFIG
from eco_assistant.utils import cleanup_vector_store


def display_section_header(title):
    """Display a section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def main():
    """Run verification tests that match notebook outputs."""
    # Clean up any existing vector store
    cleanup_vector_store()
    
    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key not found in .env file")
    print("API key loaded successfully.")
    
    # Initialize ECO Assistant
    eco_assistant = ECOAssistant(api_key=api_key, config=CONFIG)
    
    try:
        # Load documents
        display_section_header("Step 1: Loading Documents")
        documents = eco_assistant.load_documents("SYNT_DOCS")
        
        # Create vector store
        display_section_header("Step 2: Creating Vector Store")
        eco_assistant.create_vector_store()
        
        # Build QA chain with examples
        display_section_header("Step 3: Building QA Chain")
        examples = """
        Q: For ECO-100001, extract the following fields: Title, Description of Change, Reason for Change, Affected Parts, and Effective Date.
        A:
        Title: Enclosure Update – Add Ventilation Slots
        Description of Change: Ventilation slots were added to the top shell to improve thermal performance.
        Reason for Change: Improve thermal performance.
        Affected Parts: PRT-000210
        Effective Date: 2025-05-01

        Q: For ECO-100002, extract the following fields: Title, Description of Change, Reason for Change, Affected Parts, and Effective Date.
        A:
        Title: Battery Type Replacement – Lithium Polymer to Solid-State
        Description of Change: Replaced lithium-polymer battery with solid-state battery in the MatrixSync X100. BOMs and documentation updated.
        Reason for Change: Improve battery safety, increase product lifespan, and align with new supplier standards.
        Affected Parts: BAT-000011 | Battery – Li-Po | Rev A → Obsolete, BAT-000014 | Battery – Solid-State | New Part, BOM-000122 | MatrixSync X100 BOM | Updated battery component
        Effective Date: 2025-05-05
        """
        eco_assistant.build_qa_chain(examples)
        
        # Test standard Q&A functionality
        display_section_header("Test 1: Standard Q&A")
        test_question = "For ECO-100002, extract the following fields: Title, Description of Change, Reason for Change, Affected Parts, and Effective Date."
        test_result = eco_assistant.query(test_question)
        print("\nAnswer:\n", test_result["result"])
        
        # Test simple query
        display_section_header("Test 2: Simple Query")
        result = eco_assistant.query("What change was made in ECO-100002 and why?")
        print(result["result"])  

        # Test structured output
        display_section_header("Test 3: Structured Output")
        structured_result = eco_assistant.get_structured_output(
            "For ECO-100002, extract the following fields: Title, Description of Change, Reason for Change, Affected Parts, and Effective Date."
        )
        print("\nFinal Structured JSON:") 
        print(json.dumps(structured_result, indent=2, ensure_ascii=False))

        # Test batch processing
        display_section_header("Test 4: Batch Processing")
        eco_numbers = ["ECO-100001", "ECO-100002", "ECO-100003", "ECO-100004"]
        batch_results = eco_assistant.batch_process_ecos(eco_numbers)

        # Test agent routing
        display_section_header("Test 5: Agent Routing")
        structured_query = "Give me a structured JSON summary of ECO-100002"
        plain_query = "What is the change in ECO-100001?"
        print("\nStructured JSON Output:")
        print(eco_assistant.route_query(structured_query))
        print("\nNatural Language Answer:") 
        print(eco_assistant.route_query(plain_query))  

        # Test multiple queries
        display_section_header("Test 6: Multiple Queries")
        queries = [
            "What change was made in ECO-100001?",
            "Give me a structured summary of ECO-100002.",
            "For ECO-100004, extract the field Affected Parts only."
        ]
        
        for q in queries:
            print(f"\nQuery: {q}")
            try:
                if "structured" in q.lower() or "json" in q.lower():
                    print("\nStructured Output:")
                    result = eco_assistant.route_query(q)
                    print(result)
                else:
                    print("\nNatural Language Answer:")
                    result_text = eco_assistant.route_query(q)
                    lines = result_text.splitlines()
                    filtered = "\n".join([line for line in lines if line.startswith("A:")]) or lines[-1]
                    print(filtered)
        
                if queries.index(q) < len(queries) - 1:
                    time.sleep(5)  # Add delay to avoid rate limits
            except Exception as e:
                print(f"Skipping due to error: {e}") 

        # Format toggle
        display_section_header("Test 7: Format Toggle")
        user_query = "What change was made in ECO-100002 and why?"
        output_format = "json"
        json_result = eco_assistant.get_formatted_output(user_query, output_format)
        print("Response:")
        print(json.dumps(json_result, indent=2))

        # Evaluation
        display_section_header("Test 8: Answer Evaluation")
        evaluation_query = "What change was made in ECO-100002 and why?"
        gold_answer = """
        The lithium-polymer battery in the MatrixSync X100 was replaced with a solid-state lithium battery 
        to improve safety and performance. The reason for this change was not explicitly stated in the ECO.
        """
        evaluation = eco_assistant.evaluate_answer(evaluation_query, gold_answer)
        print("\nEvaluation Result:\n", evaluation["raw_result"])

        # Stakeholder email
        display_section_header("Test 9: Stakeholder Email")
        email = eco_assistant.generate_stakeholder_email("ECO-100002")
        print(email)
        
        display_section_header("Verification Complete")
        print("All tests completed.")
        
    except Exception as e:
        print(f"Error during verification: {e}")
    
    finally:
        # Clean up resources
        eco_assistant.cleanup()


if __name__ == "__main__":
    main()
