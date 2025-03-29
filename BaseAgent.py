from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# --- FAISS Memory Manager ---
class FAISSManager:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Default dimension for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.load_or_create_index()

    def load_or_create_index(self):
        """Load existing index and texts or create new ones."""
        if os.path.exists('faiss_index.pkl') and os.path.exists('texts.pkl'):
            with open('faiss_index.pkl', 'rb') as f:
                self.index = pickle.load(f)
            with open('texts.pkl', 'rb') as f:
                self.texts = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.texts = []

    def save_index(self):
        """Save index and texts to disk."""
        with open('faiss_index.pkl', 'wb') as f:
            pickle.dump(self.index, f)
        with open('texts.pkl', 'wb') as f:
            pickle.dump(self.texts, f)

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

# --- Simple Conversation Memory ---
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

# --- Base Agent ---
class BaseAgent:
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name
        self.llm = OllamaLLM(model=model_name)
        self.memory = SimpleConversationMemory()
        self.wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        self.use_memory = False
        self.faiss_db = None

    def is_factual_query(self, user_query):
        """Determine if a query is factual (basic heuristic)."""
        factual_keywords = ["what", "who", "when", "where", "capital", "history", "define", "explain"]
        return any(word in user_query.lower() for word in factual_keywords)

    def generate_response(self, user_query):
        """Generates a response using the LLM."""
        context_str = ""
        if self.use_memory and self.faiss_db:
            context = self.faiss_db.retrieve_context(user_query)
            context_str = "\n".join(context) if context else ""

        prompt_template = PromptTemplate(
            input_variables=["input"],
            template="Answer the following question: {input}\nAI:"
        )

        chain = prompt_template | self.llm
        response = chain.invoke({"input": user_query})

        if self.use_memory and self.faiss_db:
            self.faiss_db.add_to_memory(f"User: {user_query}\nAI: {response}")

        return response

# --- General Agent ---
class GeneralAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.use_memory = True
        self.faiss_db = FAISSManager()

    def handle_query(self, user_query):
        if self.is_factual_query(user_query):
            wikipedia_response = self.wikipedia_tool.run(user_query)
            prompt = f"Based on the following information: {wikipedia_response}\n\nAnswer the question: {user_query}"
        else:
            prompt = f"Answer the following general question: {user_query}"
        return self.generate_response(prompt)

# --- Admission Agent ---
class AdmissionAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.use_memory = False

    def handle_query(self, user_query):
        prompt = f"Provide admission details for Concordia: {user_query}"
        return self.generate_response(prompt)

# --- AI Agent ---
class AIAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.use_memory = False

    def handle_query(self, user_query):
        if self.is_factual_query(user_query):
            wikipedia_response = self.wikipedia_tool.run(user_query)
            prompt = f"Based on the following information: {wikipedia_response}\n\nAnswer the question: {user_query}"
        else:
            prompt = f"Answer AI-related questions: {user_query}"
        return self.generate_response(prompt)

# --- Multi-Agent Coordinator ---
class AgentCoordinator:
    def __init__(self):
        self.general_agent = GeneralAgent()
        self.admission_agent = AdmissionAgent()
        self.ai_agent = AIAgent()

    def route_query(self, user_query):
        """
        Routes the user query to the appropriate agent based on keywords.
        This is a simple heuristic-based approach and can be improved.
        """
        query_lower = user_query.lower()
        if "admission" in query_lower or "concordia" in query_lower:
            return self.admission_agent.handle_query(user_query)
        elif any(keyword in query_lower for keyword in ["ai", "machine learning", "deep learning", "unsupervised", "supervised"]):
            return self.ai_agent.handle_query(user_query)
        else:
            return self.general_agent.handle_query(user_query)

# --- Testing the Multi-Agent Coordinator ---
if __name__ == "__main__":
    coordinator = AgentCoordinator()

    # Example queries demonstrating routing
    queries = [
        "What is the capital of United States?",
        "What are the admission requirements for Canadian Quebec students?",
        "Explain what the concept of unsupervised learning is."
    ]

    for query in queries:
        print(f"\nUser Query: {query}")
        response = coordinator.route_query(query)
        print("Response:", response)