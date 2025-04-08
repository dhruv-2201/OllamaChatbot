import os
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from .utils import shared_embedding_model, save_pickle, load_pickle


class ChatbotMetrics:
    def __init__(self):
        self.embedding_model = shared_embedding_model
        self.responses: List[Dict] = []
        self.feedback_scores = []
        base_path = os.path.dirname(os.path.dirname(__file__))
        self.metrics_file = os.path.join(
            base_path, "pkl_files/chatbot_metrics.pkl")
        self.load_metrics()

    def load_metrics(self):
        """Load previous metrics if they exist"""
        saved_data = load_pickle(self.metrics_file, default={
                                 'responses': [], 'feedback_scores': []})
        self.responses = saved_data.get('responses', [])
        self.feedback_scores = saved_data.get('feedback_scores', [])

    def save_metrics(self):
        """Save metrics to file"""
        save_pickle({
            'responses': self.responses,
            'feedback_scores': self.feedback_scores
        }, self.metrics_file)

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
