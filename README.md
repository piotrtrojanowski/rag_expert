# RAG Expert Project

This project implements a simple RAG system based on three sources of information:
1. pdf
2. internet
3. llm

It requires a local installation of llm: via ollama server, ollama api, or llama.cpp + gguf.

## Structure
- `src/`: Core Python scripts for utilities, language models, and vectorstore operations.
- `notebooks/`: Original Jupyter Notebook for reference.
- `data/`: Placeholder for datasets.
- `tests/`: Placeholder for tests.

## Getting Started
1. Install dependencies from `requirements.txt`.
2. Run `src/main.py` to test the basic functionality.

## Dependencies
- PyPDF2
- langchain
- Chroma
