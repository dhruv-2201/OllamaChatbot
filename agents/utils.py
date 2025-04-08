from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
import os
import pickle
import torch

# Shared models used across multiple components
shared_embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
shared_ollama_model = OllamaLLM(model="llama3.1")

# Utility functions for file operations


def ensure_directory_exists(path):
    """Ensure the directory exists, create if it doesn't"""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_pickle(data, file_path):
    """Save data to a pickle file, ensuring directory exists"""
    ensure_directory_exists(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path, default=None):
    """Load data from a pickle file if it exists"""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return default
