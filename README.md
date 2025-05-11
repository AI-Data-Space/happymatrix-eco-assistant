# HappyMatrix ECO Assistant

A GenAI-powered tool for analyzing Engineering Change Orders.




**Author:** Olga Seymour

**Date:** May 2025

**GitHub:** https://github.com/data-ai-studio/happymatrix-eco-assistant


## Project Overview

The ECO Assistant demonstrates how Generative AI can assist engineers and product teams 
in understanding and organizing Engineering Change Orders (ECOs). It uses Google's Gemini LLMs 
combined with Retrieval-Augmented Generation (RAG) to extract, analyze, and communicate information 
from unstructured ECO documents. 

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

## Getting Started

### Prerequisites

- Python 3.10+
- Google Gemini API key (get one at [Google AI Studio](https://aistudio.google.com/app/apikey))

### Installation

1. Clone this repository:
   
   git clone https://github.com/data-ai-studio/happymatrix-eco-assistant.git
   cd happymatrix-eco-assistant
   
2. Create a virtual environment:

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install the package and dependencies: 

pip install -r requirements.txt
pip install -e .

4. Create a .env file with your Google Gemini API key:

GOOGLE_API_KEY=your_api_key_here

### Running the Demo

Run the main demo (diver) script:

python main.py

This will demonstrate core functionality: 

1. Loading and analyzing ECO documents
2. Natural language Q&A
3. Structured data extraction
4. Stakeholder email generation

For more examples:

python examples/basic_demo.py


### 📁 Script Overview

| File | Description |
|------|-------------|
| `main.py` | 🔹 **Primary driver script** — demonstrates core functionality of the ECO Assistant, including document loading, Q&A, structured output, and stakeholder email generation. |
| `examples/basic_demo.py` | 🔸 Lightweight demonstration script — a simpler version of the main demo for quick testing or reference. |
| `tests/verify_notebook_outputs.py` | ✅ Full verification script — ensures that the outputs from the structured Python code match those from the original Jupyter notebook. |
| `notebooks/ECO-assistant.ipynb` | 🧠 Original development notebook — shows the exploratory and step-by-step creation of the ECO Assistant with rich examples and documentation. |


### Testing

The project includes comprehensive testing to ensure it matches the original Jupyter notebook implementation:

python tests/verify_notebook_outputs.py

This verification script runs a complete suite of tests that reproduces all outputs from the original notebook, including:

- Document loading
- Vector storage
- Query processing
- Structured output generation
- Batch processing
- Agent routing
- Response evaluation


### Example Outputs

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

- Automatic retries with exponential backoff
- User-friendly error messages
- Strategic delays between API calls

If you encounter persistent rate limit errors when running examples:

1. Wait a few minutes before trying again
2. Run fewer operations in succession
3. Consider a paid API tier for higher limits

These limitations are standard when working with AI APIs and demonstrate real-world API integration practices.


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
├── tests/                     # Testing scripts
│   └── verify_notebook_outputs.py  # Verification tests
├── .env.example               # Template for API key
├── .gitignore                 # Git ignore file
├── main.py                    # Main demo script
├── requirements.txt           # Dependencies
├── setup.py                   # Package installation
└── README.md                  # This file
```

### About the Data

All ECO documents are synthetic and created for educational purposes. They do not reflect any real products or proprietary information.

### Author
Olga Seymour


## Acknowledgements

This project was inspired by the Google Generative AI Intensive Course and is part of a hands-on capstone project. All ECO documents are fictional and created for demonstration purposes only.

---

