# Taller RAG

This repository contains a simple Retrieval-Augmented Generation (RAG) implementation using TF-IDF for retrieval and Ollama for text generation.

## Requirements

- Python 3.6+
- scikit-learn
- ollama

Install the dependencies using:

```bash
pip install -r requirements.txt
```

You also need to install and run the llama3 model using Ollama:

```bash
ollama pull llama3
```

## Usage

Run the `main.py` script:

```bash
python main.py
```

The script will prompt you to enter a question. It will then retrieve the most relevant document from the knowledge base and use the llama3 model to generate an answer based on the retrieved document.

## Implementation

The retrieval mechanism uses TF-IDF to vectorize the documents and the user's query. Cosine similarity is then used to find the most relevant document.

The text generation is done using the llama3 model via the Ollama client.

## Evaluation

The `main.py` file contains a set of evaluation questions that can be used to test the RAG implementation.
