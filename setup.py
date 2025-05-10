"""
Setup script for the HappyMatrix ECO Assistant package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="eco-assistant",
    version="0.1.0",
    author="Olga Seymour",
    description="A GenAI-powered Engineering Change Order Analysis Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/data-ai-studio/happymatrix-eco-assistant",  
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha", 
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "langchain>=0.0.267",
        "langchain-community>=0.0.10",
        "langchain-google-genai>=0.0.3",
        "chromadb>=0.4.18",
        "google-generativeai>=0.3.1",
        "python-dotenv>=1.0.0",
        "pandas>=2.0.3",
    ],
)