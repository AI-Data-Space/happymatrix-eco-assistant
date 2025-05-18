"""
Main ECO Assistant module containing the ECOAssistant class.

This module provides the core functionality for analyzing and extracting
information from Engineering Change Order (ECO) documents using
Google's Gemini models and LangChain.

Author: Olga Seymour
Date: May 2025
Github: https://github.com/AI-Data-Space/happymatrix-eco-assistant 
"""

import json
import re
import time
import pandas as pd
import os  # for future file path handling - will implement later

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

        I've found Gemini's flash model offers the best balance of speed and quality
        for this specific RAG use case. The embedding model choice matters less since
        we're mostly dealing with technical documentation with specific terminology.
        """
        try:
            # Initialize the Gemini LLM. Create the main Gemini model instance
            self.llm = ChatGoogleGenerativeAI(
                model=self.config["model"],
                google_api_key=self.api_key
            )
            # Initialize the embedding model for vectors
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
        """Create vector store from loaded documents.

        This is typically the slowest part of the initialization process,
        especially with large document sets. If you're having performance
        issues, try reducing the chunk size in your config.

        Returns:
            Chroma: ChromaDB vector store instance

        Raises:
            ValueError: If no documents have been loaded
            RuntimeError: If embedding process fails
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

        except ImportError as e:
            # Error for missing dependencies
            print(f"Missing required dependency: {e}")
            print("Try running 'pip install chromadb langchain-text-splitters'")
            raise

        except MemoryError:
            # Handle out of memory during embedding of large docs
            print(
                "ERROR: Out of memory while creating embeddings. Try reducing chunk size or processing fewer documents.")
            raise RuntimeError("Memory limit exceeded during vector embedding")

        except Exception as e:
            # Look for common ChromaDB issues
            if "already exists" in str(e).lower():
                print(
                    f"Database at {self.config['persist_dir']} already exists. Try using a different directory or delete the existing one.")
            elif "connection" in str(e).lower():
                print("Database connection failed. Check if ChromaDB server is running.")
            else:
                print(f"Error creating vector store: {e}")
                print("For debugging: Check permissions on persist_dir and verify embedding model is correctly specified.")
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
        
        # The examples format is particularly important for consistent output
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
            # I tried 'map_reduce' chain_type first but 'stuff' works better for our use case
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

        attempt = 0
        max_attempts = 3
        backoff = 2  # seconds

        while attempt < max_attempts:
            try:
                return self.qa_chain.invoke(question)
            except Exception as e:
                attempt += 1
                if "rate limit" in str(e).lower() and attempt < max_attempts:
                    # My API key often hits rate limits around noon PST
                    wait_time = backoff * attempt
                    print(f"Rate limit hit, waiting {wait_time}s before retry {attempt}/{max_attempts}...")
                    time.sleep(wait_time)
                else:
                    print(f"Error querying: {e}")
                    return {"result": f"An error occurred: {str(e)}"}

    def get_structured_output(self, question, eco_number=None):
        """Get structured JSON output for a question.

        This method is the workhorse of the ECO Assistant - it extracts structured
        data that can be used by other systems. I've found it works best when you
        explicitly include the ECO number in your question.

        Args:
            question (str): Question about an ECO
            eco_number (str, optional): ECO number. Extracted from question if None.

        Returns:
            dict: Structured JSON with ECO fields
        """
        try:

            # Extract ECO number from question if not provided
            # Using regex to find ECO-XXXXXX pattern
            if not eco_number:
                match = re.search(r"(ECO-\d+)", question)

                if not match:
                    eco_examples = "ECO-12345, ECO-98765"
                    raise ValueError(
                        f"No ECO number found in query. Please include an ECO number in the format {eco_examples}")

            eco_number = match.group(1) if match else "Unknown"

            # Check that extracted ECO is in our database
            eco_exists = any(eco_number in doc.metadata.get("source", "") for doc in self.documents)
            if not eco_exists:
                print(f"WARNING: {eco_number} might not be in the loaded documents. Results may be incomplete.")
                print(
                    f"Available ECOs: {', '.join([d.metadata.get('source', '').split('/')[-1] for d in self.documents[:5]])}")

            # Get raw answer from QA chain using our RAG query
            result = self.query(question)

            # Use Gemini to reformat as JSON
            # This structured prompt helps ensure consistent output
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

            # Clean up any markdown formatting
            if output.startswith("```json"):
                output = output.replace("```json", "").replace("```", "").strip()

            try:
                parsed_json = json.loads(output)
                # Verification that required fields exist
                required_fields = ["ECO Number", "Title", "Description of Change"]
                missing_fields = [field for field in required_fields if field not in parsed_json]
                if missing_fields:
                    print(f"WARNING: Structured output missing fields: {', '.join(missing_fields)}")

                return parsed_json
            except json.JSONDecodeError:
                print(f"ERROR: Failed to parse JSON output for {eco_number}. Model returned invalid JSON.")
                print("First 100 characters of output:", output[:100] + "..." if len(output) > 100 else output)
                # Still try to return something useful
                return {"ECO Number": eco_number, "error": "Invalid JSON output", "raw_text": output}

        except Exception as e:
            print(f"Error getting structured output: {e}")
            return {"error": str(e)}

    def batch_process_ecos(self, eco_numbers):
        """Process multiple ECOs and return structured results.

        Args:
            eco_numbers (list): List of ECO numbers to process

        Returns:
            pandas.DataFrame: Results as a dataframe, also saves CSV and JSON files
        """
        structured_results = []
        skipped_ecos = []

        start_time = time.time()
        print(f"Starting batch processing at {time.strftime('%H:%M:%S')}")

        # Process each ECO - tracking to report completion %
        for i, eco in enumerate(eco_numbers):
            print(f"\n[{i + 1}/{len(eco_numbers)}] Processing {eco}...")

            # Rate limit protection - critical for larger batches
            if i > 0 and i % 10 == 0:
                print(f"Taking a break to avoid rate limits ({i}/{len(eco_numbers)} complete)")
                time.sleep(5)  # My API key usually needs a break after 10 requests

            # Create query for this ECO
            query = f"For {eco}, extract the following fields: Title, Description of Change, Reason for Change, Affected Parts, and Effective Date."

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

                # Error handler for JSON parsing issues
                except json.JSONDecodeError:
                    print(f"ERROR: Failed to parse JSON response for {eco}. Output format may be incorrect.")
                    print("Raw output:", output[:100] + "..." if len(output) > 100 else output)
                    skipped_ecos.append(eco)

            except Exception as e:
                print(f"Error processing {eco}: {e}")
                skipped_ecos.append(eco)

                # Save results to files
        if structured_results:
            df = pd.DataFrame(structured_results)
            df.to_csv("eco_structured_results.csv", index=False)
            with open("eco_structured_results.json", "w") as f:
                json.dump(structured_results, f, indent=2)

                # Performance summary
            elapsed = time.time() - start_time
            print("\nBatch processing complete.")
            print(
                f"Processed {len(structured_results)} ECOs in {elapsed:.1f}s ({len(structured_results) / elapsed:.2f} ECOs/second)")
            print("Structured ECO results are displayed below.")
            print("The results are also saved as CSV and JSON files.")

            # Display the dataframe if in a notebook context
            try:
                from IPython.display import display
                display(df)
            except ImportError:
                print(df)

        else:
            print("\nNo results were successfully processed.")

            # Summary of skipped items
        if skipped_ecos:
            print(f"\nWARNING: {len(skipped_ecos)}/{len(eco_numbers)} ECOs were skipped due to errors")
            if len(skipped_ecos) <= 5:
                print(f"Skipped ECOs: {', '.join(skipped_ecos)}")
            else:
                print(f"First 5 skipped ECOs: {', '.join(skipped_ecos[:5])}...")

        return df if structured_results else pd.DataFrame()

    def qa_tool(self, query):
        """Natural language answer tool.

        Returns focused answers about specific ECOs mentioned in the query.
        I've found this works best for quick questions where you don't need
        structured data.

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
        """Structured JSON output tool.

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
        """Route query to natural language or JSON format based on intent.

        For integrating with tools that need to determine which format the
        user wants automatically. Simple heuristic but works pretty well.

        Args:
            query (str): User's question

        Returns:
            str: Response in either JSON or natural language format
        """
        # Look for keywords that suggest structured output preference
        if "structured" in query.lower() or "json" in query.lower():
            return self.structured_tool(query)

        # Default to plain language for better readability
        return self.qa_tool(query)

    def process_multiple_queries(self, queries, delay=10):
        """Process multiple queries through the agent router.

        Useful for batch testing or processing a list of standard questions.

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
        """Get output in either JSON or plain text format.

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
        """Evaluate model answer against a reference.

        I use this during development to track quality improvements over time.
        It's not perfect but gives a good indication of how changes affect output.

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
        # TODO: Add option to customize email template
        """Generate a stakeholder email about an ECO.

        This is super useful for quickly creating comms that engineering managers
        can send to their teams. Saves tons of time compared to writing these by hand.

        Args:
            eco_number (str): ECO number to summarize

        Returns:
            str: Formatted email text
        """
        try:
            # Validate ECO number format
            validate_eco_number(eco_number)

            # Get ECO context from our vector DB
            eco_query = f"What is the change described in {eco_number}? Extract details to create a stakeholder email."
            eco_context = call_with_retry(lambda: self.query(eco_query))["result"]

            # Prompt for email generation with specific sections
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
        Clean up resources and safely close the ChromaDB connection.

        This method should be called when processing is complete to avoid 
        leaving open database connections. It checks for an active database 
        client and attempts to close it if a 'close' method is available.
        Any errors during cleanup are caught and reported without interrupting 
        execution.                                           
           
        """
        try:

            # Only attempt cleanup if we have a database client
            if self.db and hasattr(self.db, '_client'):
                client = self.db._client
                # Close ChromaDB connection if the method exists
                if hasattr(client, 'close'):
                    client.close()
                    print("ChromaDB connection closed")

            print("\nCleanup complete")
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")


                    
            
