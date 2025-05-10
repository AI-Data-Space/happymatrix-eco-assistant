"""
Main ECO Assistant module containing the ECOAssistant class.

This module provides the core functionality for analyzing and extracting
information from Engineering Change Order (ECO) documents using
Google's Gemini models and LangChain.
"""

import json
import re
import time
import pandas as pd


import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from eco_assistant.utils import (
    load_and_tag_documents,
    create_vector_db,
    call_with_retry,
    validate_eco_number
)


class ECOAssistant:
    """
    ECO Assistant for analyzing Engineering Change Orders.
    
    This class provides a RAG-powered assistant that can answer questions about
    Engineering Change Orders, extract structured data in JSON format, and
    generate stakeholder communications.
    
    Attributes:
        api_key (str): The Google API key for Gemini access
        config (dict): Configuration parameters for models and retrieval
        llm: The language model instance
        db: Vector database for document storage and retrieval
        qa_chain: LangChain QA chain for answering questions
        documents (list): Raw loaded ECO documents
        split_docs (list): Chunked documents after text splitting
    """
    
    def __init__(self, api_key, config=None):
        """
        Initialize the ECO Assistant.
        
        Args:
            api_key (str): Google Gemini API key
            config (dict, optional): Configuration for models and retrieval settings.
                                    Uses default CONFIG if None.
        """
        self.api_key = api_key
        self.config = config or {}
        self.llm = None
        self.db = None
        self.qa_chain = None
        self.documents = None
        self.split_docs = None
        self.embedding = None
        
        # Validate config has required fields
        required_keys = ["chunk_size", "chunk_overlap", "retriever_k", 
                         "persist_dir", "model", "embedding_model"]
        for key in required_keys:
            if key not in self.config:
                print(f"Missing key: {key}. Using default.") 
        
        self.setup()
    
    def setup(self):
        """
        Initialize the LLM and embedding models.
        
        Sets up the Gemini model for text generation and embedding model
        for vector representations. Handles errors if initialization fails.
        """
        try:
            # Initialize the Gemini LLM
            self.llm = ChatGoogleGenerativeAI(
                model=self.config["model"],
                google_api_key=self.api_key
            )
    
            # Initialize the embedding model
            self.embedding = GoogleGenerativeAIEmbeddings(
                model=self.config["embedding_model"],
                google_api_key=self.api_key
            )
        except Exception as e:
            print(f"Error initializing models: {e}")
            raise
   
    def load_documents(self, folder_path):
        """
        Load and tag ECO documents from a folder.
        
        Reads text files from the specified folder and tags each document
        with its ECO number extracted from the filename.
        
        Args:
            folder_path (str): Path to folder containing ECO text files
            
        Returns:
            list: List of tagged document texts
            
        Raises:
            FileNotFoundError: If directory doesn't exist
            NotADirectoryError: If path is not a directory
        """
        try:
            self.documents = load_and_tag_documents(folder_path)
            return self.documents
        except Exception as e:
            print(f"Error loading documents: {e}")
            raise
    
    def create_vector_store(self):
        """
        Create a vector store from loaded documents.
        
        Splits documents into chunks, creates embeddings, and stores them
        in ChromaDB for semantic search and retrieval.
        
        Returns:
            Chroma: ChromaDB vector store instance
            
        Raises:
            ValueError: If no documents have been loaded
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
            
        try:
            # Create vector DB from documents
            self.db, self.split_docs = create_vector_db(
                self.documents, 
                self.embedding, 
                self.config["persist_dir"]
            )
            print(f"\nVector store created with {len(self.split_docs)} chunks")
            return self.db
        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise 
    
    def build_qa_chain(self, examples, k=None):
        """
        Create a question-answering chain with few-shot examples.
        
        Args:
            examples (str): Few-shot examples for the prompt
            k (int, optional): Number of documents to retrieve. Uses config value if None.
            
        Raises:
            ValueError: If LLM or vector store aren't initialized
        """
        if not self.llm or not self.db:
            raise ValueError("LLM and vector store must be initialized first.")
    
        # Use config value if k not specified
        k = k or self.config.get("retriever_k", 8)
    
        # Create prompt template with examples
        template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are assisting with reviewing ECO (Engineering Change Order) documents.
    
            From the given context, extract the following fields:
            - Title
            - Description of Change
            - Reason for Change
            - Affected Parts (list of strings)
            - Effective Date (in YYYY-MM-DD format)
    
            Use your best judgment even if fields are implied or partially mentioned. 
            Do not say "Not mentioned" unless you are certain the field is missing.
    
            {examples}
    
            Context:
            {context}
    
            Q: {question}
            A:
            """.strip()
        )
    
        try:
            # Build retrieval QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.db.as_retriever(search_kwargs={"k": k}),
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": template.partial(examples=examples)
                }
            )
            
        except Exception as e:
            print(f"Error building QA chain: {e}")
            raise
    
    def query(self, question):
        """
        Run a query through the QA chain.
        
        Args:
            question (str): The question to answer about ECO documents
            
        Returns:
            dict: Result with answer and source documents
            
        Raises:
            ValueError: If QA chain not initialized
        """
        if not self.qa_chain:
            raise ValueError("No QA chain built. Call build_qa_chain() first.")
            
        try:
            # Use retry wrapper for API rate limits
            return call_with_retry(lambda: self.qa_chain.invoke(question))
        except Exception as e:
            print(f"Error querying: {e}")
            return {"result": f"An error occurred: {str(e)}"}
    
    def get_structured_output(self, question, eco_number=None):
        """
        Get structured JSON output for a question.
        
        Args:
            question (str): Question about an ECO
            eco_number (str, optional): ECO number. Extracted from question if None.
            
        Returns:
            dict: Structured JSON with ECO fields
        """
        try:
            # Get raw answer from QA chain
            result = self.query(question)
    
            # Extract ECO number from question if not provided
            if not eco_number:
                match = re.search(r"(ECO-\d+)", question)
                eco_number = match.group(1) if match else "Unknown"
    
            # Format result as JSON using Gemini
            prompt = f"""
            You are an assistant converting ECO answers into structured JSON.
    
            Return a JSON object with the following fields:
            - ECO Number
            - Title
            - Description of Change
            - Reason for Change
            - Affected Parts (list of strings)
            - Effective Date
    
            Always use this ECO Number: {eco_number}
    
            Answer:
            {result['result']}
            """
            response = call_with_retry(lambda: self.llm.invoke(prompt))
            output = response.content.strip()
    
            # Remove markdown code block formatting if present
            if output.startswith("```json"):
                output = output.replace("```json", "").replace("```", "").strip()
    
            return json.loads(output)
    
        except Exception as e:
            print(f"Error getting structured output: {e}")
            return {"error": str(e)} 
    
    def batch_process_ecos(self, eco_numbers):
        """
        Process multiple ECOs and return structured results.
        
        Args:
            eco_numbers (list): List of ECO numbers to process
            
        Returns:
            pandas.DataFrame: Results as a dataframe, also saves CSV and JSON files
        """
        structured_results = []
    
        # Create queries for each ECO
        queries = [
            f"For {eco}, extract the following fields: Title, Description of Change, Reason for Change, Affected Parts, and Effective Date."
            for eco in eco_numbers
        ]
    
        # Process each query
        for query in queries:
            print(f"\nRunning query: {query}")
    
            try:
                # Get raw answer
                result = call_with_retry(lambda: self.qa_chain.invoke(query))
                answer = result["result"]
                print("Raw answer:", answer)
    
                # Extract ECO number from query
                eco_match = re.search(r"(ECO-\d+)", query)
                eco_number = eco_match.group(1) if eco_match else "Unknown"
    
                # Create structured JSON from answer
                structured_prompt = f"""
                You are an assistant that converts ECO answers into structured JSON.
    
                Return a JSON object with the following fields:
                - ECO Number
                - Title
                - Description of Change
                - Reason for Change
                - Affected Parts (as a list of strings)
                - Effective Date
    
                Use your best judgment to include information, even if it is implied.
                - If a part number or product is mentioned in the answer (even without labels), include it in Affected Parts.
                - If a specific Effective Date is not listed but a Date Issued is provided, use that as the Effective Date.
                - Only return "Unknown" or an empty list if there is truly no way to infer the information.
    
                Always use this ECO Number: {eco_number}
    
                Answer:
                {answer}
                """
                response = call_with_retry(lambda: self.llm.invoke(structured_prompt))
                output = response.content.strip()
    
                # Clean up markdown formatting 
                if output.startswith("```json"):
                    output = output.replace("```json", "").replace("```", "").strip()
    
                # Skip empty outputs
                if output in ("{}", "", "[]"):
                    continue
    
                # Parse JSON response
                try:
                    parsed = json.loads(output)
                    structured_results.append(parsed)
                    print("Structured successfully.")
                   
                except Exception as e:
                    print("Could not parse JSON. Skipping this item.")
                    print("Error:", e)
                    
            except Exception as e:
                print(f"Error processing query: {e}")
               
        # Save results to files
        df = pd.DataFrame(structured_results)
        df.to_csv("eco_structured_results.csv", index=False)
        with open("eco_structured_results.json", "w") as f:
            json.dump(structured_results, f, indent=2)

        print("\nBatch processing complete.")
        print("Structured ECO results are displayed below.")
        print("The results are also saved as CSV and JSON files.")

        # Display the dataframe if in a notebook context
        try:
            from IPython.display import display
            display(df)
        except ImportError:
            print(df)

        return df
    
    def qa_tool(self, query):
        """
        Natural language answer tool.
        
        Returns plain text answers focused on the specific ECO in the query.
        
        Args:
            query (str): User's question about an ECO
            
        Returns:
            str: Natural language answer
        """
        # Create a prompt that focuses on the specific ECO
        focused_prompt = f"""
        You are an assistant answering engineering questions based only on the ECO that matches the user's query.
        
        Only summarize the ECO number explicitly mentioned in the question. Ignore any unrelated ECOs.
        
        Question: {query}
        """
        try:
            return call_with_retry(lambda: self.qa_chain.invoke(focused_prompt))["result"]
        except Exception as e:
            print(f"Error in qa_tool: {e}")
            return f"Error processing query: {str(e)}"
    
    def structured_tool(self, query):
        """
        Structured JSON output tool.
        
        Returns answers as formatted JSON for integration with other systems.
        
        Args:
            query (str): User's question about an ECO
            
        Returns:
            str: JSON-formatted string with structured data
        """
        try:
            # Get context from QA chain
            context = call_with_retry(lambda: self.qa_chain.invoke(query))["result"]
            
            # Format as structured JSON
            structured_prompt = f"""
            You are reviewing ECO content and converting it into structured JSON.
            
            Extract the following fields and return a JSON object with:
            - ECO Number
            - Title
            - Description of Change
            - Reason for Change
            - Affected Parts (as a list of strings)
            - Effective Date
            
            If any field is missing or not mentioned, use "Unknown" or [].
            
            Answer:
            {context}
            """
            response = call_with_retry(lambda: self.llm.invoke(structured_prompt))
            output = response.content.strip()
            
            # Clean up markdown formatting if present
            if output.startswith("```json"):
                output = output.replace("```json", "").replace("```", "").strip()
            
            return output
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def route_query(self, query):
        """
        Route the query to the appropriate tool based on intent.
        
        Automatically detects if the user wants structured JSON or natural language.
        
        Args:
            query (str): User's question
            
        Returns:
            str: Response in either JSON or natural language format
        """
        # Check if query mentions structured output or JSON
        if "structured" in query.lower() or "json" in query.lower():
            return self.structured_tool(query)
        
        # Default to natural language
        return self.qa_tool(query)
    
    def process_multiple_queries(self, queries, delay=10):
        """
        Process multiple queries through the agent router.
        
        Args:
            queries (list): List of query strings
            delay (int): Seconds to wait between queries to avoid rate limits
            
        Returns:
            list: Results for each query with type and content
        """
        results = []
        
        for q in queries:
            print(f"Processing query: {q}")
            
            try:
                # Route based on query content
                if "structured" in q.lower() or "json" in q.lower():
                    print("Using structured output tool")
                    result = {"type": "structured", "content": self.structured_tool(q)}
                else:
                    print("Using natural language tool")
                    result_text = self.qa_tool(q)
                    
                    # Filter to show only the final answer line
                    lines = result_text.splitlines()
                    filtered = "\n".join([line for line in lines if line.startswith("A:")]) or lines[-1]
                    result = {"type": "natural_language", "content": filtered}
                    
                results.append(result)
                
                # Add a delay between queries to avoid rate limits
                if delay > 0 and queries.index(q) < len(queries) - 1:
                    print(f"Waiting {delay} seconds before next query...")
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"Error processing query: {e}")
                results.append({"type": "error", "content": str(e)}) 
        
        return results 
    
    def get_formatted_output(self, query, output_format="json"):
        """
        Get output in either JSON or plain text format.
        
        Args:
            query (str): User's question
            output_format (str): Either "json" or "text"
            
        Returns:
            dict or str: Response in requested format
        """
        print(f"Getting {output_format} output for query: {query}")
        
        # Run the query using RAG
        try:
            result = call_with_retry(lambda: self.qa_chain.invoke(query)) 
            
            # Choose prompt style based on output format
            if output_format == "json":
                prompt = f"""
                You are reviewing ECO data and returning a structured JSON object.
                
                Return these fields:
                - ECO Number
                - Title
                - Description of Change
                - Reason for Change
                - Affected Parts
                - Effective Date
                
                Context:
                {result['result']}
                
                Question: {query}
                """
            else:
                prompt = f"""
                You are reviewing an ECO and responding in plain language.
                
                Context:
                {result['result']}
                
                Question: {query}
                """
            
            # Run Gemini on the final prompt
            final_response = call_with_retry(lambda: self.llm.invoke(prompt))
            response_text = final_response.content.strip()
            
            # Clean up response formatting
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.strip("`").strip()
            
            # Parse or return based on format
            if output_format == "json":
                try:
                    return json.loads(response_text)
                except Exception as e:
                    print(f"Could not parse JSON: {e}")
                    return {"error": str(e), "raw_text": response_text}
            else:
                return response_text
                
        except Exception as e:
            print(f"Error getting formatted output: {e}")
            return {"error": str(e)} if output_format == "json" else f"Error: {str(e)}"
    
    def evaluate_answer(self, query, reference_answer):
        """
        Evaluate assistant's answer against a reference.
        
        Uses Gemini to compare the assistant's answer to a reference answer
        and provide a numerical score and justification.
        
        Args:
            query (str): The question to evaluate
            reference_answer (str): Gold-standard reference answer
            
        Returns:
            dict: Evaluation results with score and reason
        """
        print(f"Evaluating answer for query: {query}") 
        
        try:
            # Get the assistant's answer
            model_answer = call_with_retry(lambda: self.qa_chain.invoke(query))["result"]
            
            # Build evaluation prompt
            prompt = f"""
            You are reviewing an assistant's response.
            
            Compare it to the reference answer below and rate the accuracy on a scale from 1 to 5.
            
            Reference:
            {reference_answer.strip()}
            
            Assistant:
            {model_answer.strip()}
            
            Respond with:
            Score: X
            Reason: (short explanation)
            """
            
            # Get evaluation from Gemini
            response = call_with_retry(lambda: self.llm.invoke(prompt))
            result_text = response.content.strip()
            
            # Parse the evaluation result
            score_match = re.search(r"Score:\s*(\d+)", result_text)
            reason_match = re.search(r"Reason:\s*(.*)", result_text)
            
            score = int(score_match.group(1)) if score_match else None
            reason = reason_match.group(1) if reason_match else "No reason provided"
            
            evaluation = {
                "query": query,
                "reference_answer": reference_answer,
                "model_answer": model_answer,
                "score": score,
                "reason": reason,
                "raw_result": result_text
            }
            
            print(f"Evaluation complete. Score: {score}/5")
            return evaluation
            
        except Exception as e:
            print(f"Error evaluating answer: {e}") 
            return {
                "query": query,
                "reference_answer": reference_answer,
                "model_answer": "Error retrieving model answer",
                "error": str(e)
            }
    
    def generate_stakeholder_email(self, eco_number):
        """
        Generate a stakeholder email summarizing an ECO.
        
        Creates a professional email that could be sent to stakeholders
        about an engineering change, including what's changing, why,
        and what parts are affected.
        
        Args:
            eco_number (str): ECO number to summarize
            
        Returns:
            str: Formatted email text
        """
        try:
            # Validate ECO number format
            validate_eco_number(eco_number)
            
            # Get ECO context
            eco_query = f"What is the change described in {eco_number}? Extract details to create a stakeholder email."
            eco_context = call_with_retry(lambda: self.query(eco_query))["result"]
            
            # Generate email with structured format
            email_prompt = f"""
            You are writing an internal stakeholder email summarizing an engineering change (ECO).
            
            Include the following:
            - Intro: "This change notification summarizes the engineering update described in {eco_number}."
            - Summary of what's changing and why
            - Affected parts or documents
            - Who should review or approve
            - Any risks or expected delays
            - Mention if other areas are unaffected
            
            Context:
            {eco_context}
            
            Email:
            """
            
            email_response = call_with_retry(lambda: self.llm.invoke(email_prompt))
            return email_response.content.strip()
        except Exception as e:
            print(f"\nError generating stakeholder email: {e}") 
            return f"Error generating email: {str(e)}"
    
    def cleanup(self):
        """
        Clean up resources when done.
        
        Closes database connections to prevent resource leaks.
        """
        try:
            if hasattr(self, 'db') and self.db is not None:
                # Close ChromaDB connection if possible
                if hasattr(self.db, '_client') and hasattr(self.db._client, 'close'):
                    self.db._client.close()
                    
            print("\nResources cleaned up")
        except Exception as e:
            print(f"Error during cleanup: {e}") 