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
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import torch

shared_embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
shared_ollama_model = OllamaLLM(model="llama3.1")

# --- FAISS Memory Manager ---
class FAISSManager:
    def __init__(self):
        self.embedding_model = shared_embedding_model
        self.dimension = 384  # Default dimension for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.load_or_create_index()

    def load_or_create_index(self):
        base_path = os.path.dirname(__file__)  # Get the directory of the script
        faiss_path = os.path.join(base_path, "faiss_index.pkl")
        texts_path = os.path.join(base_path, "texts.pkl")

        if os.path.exists(faiss_path) and os.path.exists(texts_path):
            with open(faiss_path, 'rb') as f:
                self.index = pickle.load(f)
            with open(texts_path, 'rb') as f:
                self.texts = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.texts = []

    def save_index(self):
        base_path = os.path.dirname(__file__)  # Get script's directory
        faiss_path = os.path.join(base_path, "faiss_index.pkl")
        texts_path = os.path.join(base_path, "texts.pkl")

        with open(faiss_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(texts_path, 'wb') as f:
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
    def __init__(self):
        self.llm = shared_ollama_model
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
            prompt = (
                f"Using the context below, answer concisely and directly.\n\n"
                f"Context:\n{wikipedia_response}\n\n"
                f"Question: {user_query}\nAnswer:"
)

        else:
            prompt = (
                f"You are a helpful assistant. Answer the following question clearly and directly, "
                f"without repeating the question.\n\n"
                f"Question: {user_query}\nAnswer:"
)

        return self.generate_response(prompt)

# --- Admission Agent ---
class AdmissionAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.use_memory = False

    def handle_query(self, user_query):
        prompt = (
            f"You are an admission advisor for Concordia University's Computer Science program. "
            f"Answer the following question clearly and helpfully. Keep your response informative and direct.\n\n"
            f"Question: {user_query}\nAnswer:"
)

        return self.generate_response(prompt)

# --- AI Agent ---
class AIAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.use_memory = False

    def handle_query(self, user_query):
        if self.is_factual_query(user_query):
            wikipedia_response = self.wikipedia_tool.run(user_query)
            prompt = (
                f"Using the context below, answer concisely and directly'.\n\n"
                f"Context:\n{wikipedia_response}\n\n"
                f"Question: {user_query}\nAnswer:"
)

        else:
            prompt = (
                f"You are an expert in artificial intelligence. Provide a clear and concise response to the AI-related question below. "
                f"Do not repeat the question. Just answer directly.\n\n"
                f"Question: {user_query}\nAnswer:"
)

        return self.generate_response(prompt)

# --- Multi-Agent Coordinator ---
class AgentCoordinator:
    def __init__(self):
        self.general_agent = GeneralAgent()
        self.admission_agent = AdmissionAgent()
        self.ai_agent = AIAgent()
        self.metrics = ChatbotMetrics()  # Add metrics
        self.rl = ReinforcementLearner()
        self.last_interaction = None
        _ = shared_ollama_model.invoke("Say hello")

    def route_query(self, user_query):
        query_lower = user_query.lower()
        
        # Get base scores from keyword matching
        scores = {
            'admission': 1.0 if any(word in query_lower 
                                  for word in ['admission', 'concordia']) else 0.0,
            'ai': 1.0 if any(word in query_lower 
                            for word in ['ai', 'machine learning', 'deep learning']) else 0.0,
            'general': 0.3  # Default score for general agent
        }
        
        # Combine with RL confidence scores
        rl_scores = self.rl.get_agent_scores(query_lower)
        for agent in scores:
            scores[agent] = 0.6 * scores[agent] + 0.4 * rl_scores[agent]
        
        # Select agent with highest combined score
        agent_type = max(scores.items(), key=lambda x: x[1])[0]
        
        # Get response from selected agent
        if agent_type == 'admission':
            response = self.admission_agent.handle_query(user_query)
        elif agent_type == 'ai':
            response = self.ai_agent.handle_query(user_query)
        else:
            response = self.general_agent.handle_query(user_query)
        
        # Store for RL update
        self.last_interaction = {
            'agent_type': agent_type,
            'response': response
        }
        
        # Log the interaction for metrics
        self.metrics.log_interaction(user_query, response, agent_type)
        return response
    
    def add_user_feedback(self, score: int):
        """Add user feedback and update RL model"""
        self.metrics.add_user_feedback(score)
        
        if self.last_interaction:
            # Get latest metrics
            metrics = self.metrics.get_metrics()
            
            # Get coherence for the agent that handled the last interaction
            agent_type = self.last_interaction['agent_type']
            coherence = metrics['agent_coherence'].get(agent_type, 0.5)
            
            # Update RL model
            self.rl.update(agent_type, score, coherence)

# --- Chatbot Metrics ---
class ChatbotMetrics:
    def __init__(self):
        self.embedding_model = shared_embedding_model
        self.responses: List[Dict] = []
        self.feedback_scores = []
        self.metrics_file = os.path.join(os.path.dirname(__file__), "chatbot_metrics.pkl")
        self.load_metrics()
        
    def load_metrics(self):
        """Load previous metrics if they exist"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'rb') as f:
                saved_data = pickle.load(f)
                self.responses = saved_data.get('responses', [])
                self.feedback_scores = saved_data.get('feedback_scores', [])
                
    def save_metrics(self):
        """Save metrics to file"""
        with open(self.metrics_file, 'wb') as f:
            pickle.dump({
                'responses': self.responses,
                'feedback_scores': self.feedback_scores
            }, f)
            
    def add_user_feedback(self, score: int):
        """Add user satisfaction score (1-5)"""
        self.feedback_scores.append(score)
        self.save_metrics()  # Save after each feedback
        
    def log_interaction(self, query: str, response: str, agent_type: str):
        """Log each interaction for metric calculation"""
        self.responses.append({
            'query': query,
            'response': response,
            'agent_type': agent_type
        })
        self.save_metrics()  # Save after each interaction
    
    def calculate_coherence(self, response1: str, response2: str) -> float:
        """Calculate semantic coherence between two responses using cosine similarity"""
        emb1 = self.embedding_model.encode([response1])[0]
        emb2 = self.embedding_model.encode([response2])[0]
        return float(cosine_similarity([emb1], [emb2])[0][0])
    
    def get_metrics(self) -> Dict:
        metrics = {
            'total_interactions': len(self.responses),
            'agent_distribution': {},
            'avg_user_satisfaction': 0,
            'agent_coherence': {}
        }

        # Group responses by agent type
        agent_responses = {}
        for response in self.responses:
            agent = response['agent_type']
            if agent not in agent_responses:
                agent_responses[agent] = []
            agent_responses[agent].append(response['response'])

        # Calculate agent distribution
        for agent in agent_responses:
            metrics['agent_distribution'][agent] = len(agent_responses[agent])

        # Calculate average user satisfaction
        if self.feedback_scores:
            metrics['avg_user_satisfaction'] = np.mean(self.feedback_scores)

        # Calculate coherence within each agent's responses
        for agent, responses in agent_responses.items():
            if len(responses) > 1:
                coherence_scores = [
                    self.calculate_coherence(responses[i], responses[i + 1])
                    for i in range(len(responses) - 1)
                ]
                metrics['agent_coherence'][agent] = np.mean(coherence_scores)

        return metrics

# --- Reinforcement Learner ---
class ReinforcementLearner:
    def __init__(self):
        self.learning_rate = 0.1
        self.agent_weights = {
            'general': {'confidence': 0.5, 'rewards': []},
            'admission': {'confidence': 0.5, 'rewards': []},
            'ai': {'confidence': 0.5, 'rewards': []}
        }
        self.weights_file = os.path.join(os.path.dirname(__file__), "agent_weights.pkl")
        self.load_weights()
    
    def load_weights(self):
        """Load saved weights if they exist"""
        if os.path.exists(self.weights_file):
            with open(self.weights_file, 'rb') as f:
                self.agent_weights = pickle.load(f)
    
    def save_weights(self):
        """Save weights to file"""
        with open(self.weights_file, 'wb') as f:
            pickle.dump(self.agent_weights, f)
    
    def update(self, agent_type: str, feedback: float, coherence: float):
        try:
            # Normalize feedback to 0-1 range
            normalized_feedback = feedback / 5.0
            
            # Calculate reward as weighted sum of feedback and coherence
            reward = 0.7 * normalized_feedback + 0.3 * coherence
            
            # Update agent confidence using exponential moving average
            current = self.agent_weights[agent_type]
            current['rewards'].append(reward)
            
            # Update confidence score
            if len(current['rewards']) > 0:
                current['confidence'] = (1 - self.learning_rate) * current['confidence'] + \
                                    self.learning_rate * np.mean(current['rewards'][-5:])
            
            self.save_weights()
            
        finally:
            # Clear GPU cache regardless of whether the update succeeded or failed
            if torch.backends.mps.is_available():  # For Apple Metal (M1/M2)
                torch.mps.empty_cache()
    
    def get_agent_scores(self, query: str) -> Dict[str, float]:
        """Get current confidence scores for each agent"""
        return {agent: data['confidence'] 
                for agent, data in self.agent_weights.items()}

# --- Testing the Multi-Agent Coordinator ---
if __name__ == "__main__":
    coordinator = AgentCoordinator()
    print("Chatbot initialized. Type 'quit' to exit.\n")

    while True:
        # Get user query
        user_query = input("\nEnter your question: ").strip()
        
        # Check for exit condition
        if user_query.lower() == 'quit':
            break
            
        # Get and display response
        response = coordinator.route_query(user_query)
        print("\nResponse:", response)
        
        # Get user feedback
        while True:
            try:
                feedback = int(input("\nPlease rate this response (1-5, where 5 is best): "))
                if 1 <= feedback <= 5:
                    coordinator.add_user_feedback(feedback)
                    break
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")