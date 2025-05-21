# HappyMatrix ECO Assistant

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-stable-green.svg)

A GenAI-powered tool for analyzing Engineering Change Orders.




**Author:** Olga Seymour

**Date:** May 2025

**GitHub:** https://github.com/AI-Data-Space/happymatrix-eco-assistant 


## Project Overview

The ECO Assistant demonstrates how Generative AI can assist engineers and product teams 
in understanding and organizing Engineering Change Orders (ECOs). It uses Google's Gemini LLMs 
combined with Retrieval-Augmented Generation (RAG) to extract, analyze, and communicate information 
from unstructured ECO documents. 

### Why This Matters
Engineering Change Orders are critical documents in product development that are often:
- Time-consuming to analyze manually
- Inconsistent in format and structure
- Difficult to integrate into downstream systems

This assistant demonstrates how GenAI can transform unstructured ECO documents into actionable insights and structured data, saving engineering teams valuable time and improving decision-making.


## Features

- **Natural Language Q&A**: Query documents using plain language
- **Structured Data Extraction**: Convert unstructured text into JSON
- **Semantic Search**: Find relevant ECO content based on meaning
- **Stakeholder Communication**: Auto-generate email summaries
- **Few-Shot Learning**: Improve extraction with examples
- **Batch Processing**: Process multiple ECOs at once

## Technologies

- **Google Gemini 1.5 Flash** - Large Language Model
- **Retrieval Augmented Generation (RAG)** - Core technique for document Q&A
- **LangChain** - Orchestration framework
- **ChromaDB** - Vector database for semantic search
- **Python** - Implementation language

## Technical Architecture

The ECO Assistant uses a multi-stage pipeline:

1. **Document Processing**: ECO documents are loaded, tagged with their identifiers, and split into chunks
2. **Vector Embedding**: Text chunks are converted to vector embeddings using Gemini's embedding model
3. **Semantic Search**: ChromaDB enables retrieval of the most relevant document chunks for each query
4. **Context-Enhanced Generation**: Retrieved context is sent to Gemini along with the query and few-shot examples
5. **Format Control**: Outputs are processed into either natural language or structured JSON based on user preference

This Retrieval-Augmented Generation (RAG) approach grounds all responses in the actual ECO document content, ensuring accuracy while leveraging Gemini's language capabilities.


### Project Structure

```
happymatrix-eco-assistant/
├── eco_assistant/             # Main package
│   ├── __init__.py            # Package initialization
│   ├── assistant.py           # ECOAssistant class
│   ├── utils.py               # Helper functions
│   └── config.py              # Configuration settings
├── examples/                  # Example scripts
│   └── basic_demo.py          # Simple demo
├── notebooks/                 # Jupyter notebooks
│   └── ECO-assistant.ipynb    # Original development notebook
├── SYNT_DOCS/                 # Synthetic ECO documents             
├── .env.example               # Template for API key
├── .gitignore                 
├── main.py                    # Main demo script
├── requirements.txt           # Dependencies
├── setup.py                   # Package installation
└── README.md                  # Project documentation
```


### 📁 Script Overview

| File | Description |
|------|-------------|
| `main.py` | 🔹 Primary demo script — runs a quick demonstration of the ECO Assistant's core capabilities, including document loading, simple Q&A, structured JSON output, and stakeholder email generation. |
| `eco_assistant/__init__.py` | Package initialization — defines package version, imports, and author information. |
| `eco_assistant/assistant.py` | ECOAssistant class — core implementation containing all functionality for analyzing Engineering Change Orders using RAG and Gemini. |
| `eco_assistant/utils.py` | Helper functions — utilities for document loading, vector database management, and API retry logic. |
| `eco_assistant/config.py` | Configuration settings — default parameters for models, chunking, and retrieval options. |
| `examples/basic_demo.py` | 🔸 Lightweight demonstration script — a simpler version of the main demo focused on basic Q&A and document loading, with more explicit path handling. |
| `notebooks/ECO-assistant.ipynb` | 🧠 Original development notebook — shows the exploratory and step-by-step creation of the ECO Assistant with detailed explanations and output examples. |
| `setup.py` | Package installation — configures package metadata and dependencies for installation. |
| `requirements.txt` | Dependencies — lists all required Python packages needed to run the assistant. |
| `.env.example` | API key template — shows how to structure the API key environment variables. |
| `README.md` | Project documentation — provides overview, installation instructions, and usage examples. |


## Getting Started

### Prerequisites

- Python 3.10+
- Google Gemini API key (get one at [Google AI Studio](https://aistudio.google.com/app/apikey))

### Installation

1. Clone this repository:      
   ```bash
   git clone https://github.com/AI-Data-Space/happymatrix-eco-assistant.git
   cd happymatrix-eco-assistant
   
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install the package and dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .

4. Create a .env file with your Google Gemini API key:
   ```bash
   GOOGLE_API_KEY=your_api_key_here

### API Key Security

⚠️ **IMPORTANT**: Never commit your `.env` file with your actual API key to version control. 
The `.env` file is listed in `.gitignore` and should remain private to your local development environment.

### Running the Demo

Run the main demo script:

python main.py

This will demonstrate core functionality: 

1. Loading and analyzing ECO documents
2. Natural language Q&A
3. Structured data extraction
4. Stakeholder email generation

For more examples:

python examples/basic_demo.py


### Quick Start

from eco_assistant import ECOAssistant

#### Initialize with your API key
assistant = ECOAssistant(api_key="your-api-key")

#### Load documents
assistant.load_documents("path/to/docs")
assistant.create_vector_store()

#### Ask a question
result = assistant.query("What change was made in ECO-100002?")
print(result["result"])


### Example Outputs
Here are examples of what you can expect when running the assistant:

## Natural Language Q&A

```
Q: What change was made in ECO-100002 and why?
A: In ECO-100002, the lithium-polymer battery in the MatrixSync X100 was 
   replaced with a solid-state battery. This was done to improve battery 
   safety, increase product lifespan, and align with new supplier standards.
```
   
## Structured JSON Output

```json
{
  "ECO Number": "ECO-100002",
  "Title": "Battery Type Replacement – Lithium Polymer to Solid-State",
  "Description of Change": "Replaced lithium-polymer battery with solid-state battery in the MatrixSync X100.",
  "Reason for Change": "Improve battery safety, increase product lifespan, and align with new supplier standards.",
  "Affected Parts": [
    "BAT-000011 | Battery – Li-Po | Rev A → Obsolete",
    "BAT-000014 | Battery – Solid-State | New Part",
    "BOM-000122 | MatrixSync X100 BOM | Updated battery component"
  ],
  "Effective Date": "2025-05-05"
}
```

### ⚠️ API Rate Limits

This project uses the Google Gemini API which has usage limits on the free tier. The code includes built-in handling for rate limits, including:

* Automatic retries with exponential backoff
* User-friendly error messages 
* Strategic delays between API calls

If you encounter persistent rate limit errors when running examples:

1. Wait a few minutes before trying again
2. Run fewer operations in succession
3. Run only a subset of ECOs in batch processing examples
4. Consider a paid API tier for higher limits

These limitations are standard when working with AI APIs and demonstrate real-world API integration practices. The notebook is designed to handle these limitations gracefully while still showing the full capability of the assistant.

> **Note**: If you plan to use this code in production, implementing a caching layer for embeddings and responses would significantly reduce API calls and improve performance.


## Jupyter Notebook

This project evolved from a Jupyter notebook where I explored and developed the core concepts.
The notebook contains:
- Detailed exploration of the RAG implementation
- Step-by-step development of the ECO Assistant
- Visualizations and output examples
- Comprehensive documentation of the approach

I've included the original notebook in this repository to show my development process and provide additional context 
for how the project was created. The packaged Python code in this repository is a refined, production-ready implementation 
of the concepts developed in the notebook.            

To explore the development process:
   
   jupyter notebook notebooks/ECO-assistant.ipynb 


## About the Data

All ECO documents included in this project are synthetic and were created solely for educational purposes. The documents are located in the `SYNT_DOCS` folder and represent fictional engineering change orders for the "MatrixSync X100" fitness tracker - a product that doesn't exist.

These synthetic documents demonstrate common patterns found in engineering change management but do not reflect any real products, companies, or proprietary information. They showcase various types of engineering changes including:

- Hardware modifications
- Component replacements
- Material changes
- Firmware updates

The synthetic nature of these documents makes this project suitable for educational use without concerns about intellectual property or confidential information.

### Author
Olga Seymour


## Acknowledgements

This project builds upon foundation concepts and patterns learned from the Google Generative AI with Gemini API course on Kaggle. Specifically, the following implementation patterns were adapted from the course labs:

- Basic Gemini API integration and configuration
- RAG implementation framework using ChromaDB vector storage
- Few-shot prompting techniques for consistent extraction
- Structured output generation using LangChain

I've extended these concepts to create a specialized application for Engineering Change Order (ECO) processing, adding:
- Domain-specific prompting for ECO document understanding
- Comprehensive error handling and rate limit management
- Structured data extraction pipeline for ECO metadata
- Email generation for stakeholder communications
- Batch processing capabilities for multiple ECOs

The course provided an excellent learning foundation that was then applied to this specialized domain to solve real-world engineering documentation challenges.
All ECO documents are fictional and created for demonstration purposes only. 

---


