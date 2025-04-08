import os
import faiss
import numpy as np
from typing import List
from .utils import shared_embedding_model, save_pickle, load_pickle


class FAISSManager:
    def __init__(self):
        self.embedding_model = shared_embedding_model
        self.dimension = 384  # Default dimension for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.load_or_create_index()

    def load_or_create_index(self):
        # Get the directory of the script
        base_path = os.path.dirname(os.path.dirname(__file__))
        faiss_path = os.path.join(base_path, "pkl_files/faiss_index.pkl")
        texts_path = os.path.join(base_path, "pkl_files/texts.pkl")

        if os.path.exists(faiss_path) and os.path.exists(texts_path):
            with open(faiss_path, 'rb') as f:
                self.index = pickle.load(f)
            with open(texts_path, 'rb') as f:
                self.texts = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.texts = []

    def save_index(self):
        base_path = os.path.dirname(os.path.dirname(__file__))
        faiss_path = os.path.join(base_path, "pkl_files/faiss_index.pkl")
        texts_path = os.path.join(base_path, "pkl_files/texts.pkl")

        save_pickle(self.index, faiss_path)
        save_pickle(self.texts, texts_path)

    def add_to_memory(self, text):
        """Add text to FAISS index with embeddings."""
        embedding = self.embedding_model.encode([text])[0]
        self.index.add(np.array([embedding]).astype('float32'))
        self.texts.append(text)
        self.save_index()

    def retrieve_context(self, query, top_k=3):
        """Retrieve relevant context based on query."""
        if len(self.texts) == 0:
            return []

        query_embedding = self.embedding_model.encode([query])[0]
        top_k = min(top_k, len(self.texts))

        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'),
            top_k
        )

        return [self.texts[i] for i in indices[0]]


class SimpleConversationMemory:
    def __init__(self):
        self.chat_history = []

    def load_memory_variables(self, inputs):
        # Returns the conversation history as a string
        return {"chat_history": "\n".join(self.chat_history)}

    def save_context(self, inputs, outputs):
        # Append the latest user input and LLM output to history
        self.chat_history.append(f"User: {inputs.get('input')}")
        self.chat_history.append(f"AI: {outputs}")
