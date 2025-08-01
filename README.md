# NeuroEmployee-RAG-Chatbot
This repository contains the NeuroEmployee service, an example of a RAG (Retrieval-Augmented Generation) application designed to answer questions based on a loaded document database.

What the Project Does:
Loads documents from the docs/ folder (or uses demo texts).
Builds a FAISS vector index using SentenceTransformer embeddings.
Finds relevant context for user queries and generates responses using a selected LLM.
Implements filtering of queries to detect confidential data.
Checks generated responses for hallucinations (by comparing embeddings) and length.
Collects query metrics and latency data through Prometheus.

Main Purpose:
Demonstrates a production approach to a RAG bot with security and monitoring.
Allows for the selection of different LLM models and embeddings through aliases.

Security:
Blocks queries containing personal or financial data (such as passwords, passport details, or SNILS) to prevent leaks.
Hallucinations: Filters out responses whose vector similarity to the context is below 0.5 or whose length is fewer than 10 words.

Safety:
Ensures the bot does not become an unofficial "storage" of personal information.
The hallucination detector makes the RAG system more reliable: it either responds with factual information from the database or honestly admits it doesn't know.
