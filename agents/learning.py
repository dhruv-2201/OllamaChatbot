import os
import numpy as np
from typing import Dict
from .utils import save_pickle, load_pickle


class ReinforcementLearner:
    def __init__(self):
        self.learning_rate = 0.1
        self.agent_weights = {
            'general': {'confidence': 0.5, 'rewards': []},
            'admission': {'confidence': 0.5, 'rewards': []},
            'ai': {'confidence': 0.5, 'rewards': []}
        }
        base_path = os.path.dirname(os.path.dirname(__file__))
        self.weights_file = os.path.join(
            base_path, "pkl_files/agent_weights.pkl")
        self.load_weights()

    def load_weights(self):
        """Load saved weights if they exist"""
        weights = load_pickle(self.weights_file)
        if weights:
            self.agent_weights = weights

    def save_weights(self):
        """Save weights to file"""
        save_pickle(self.agent_weights, self.weights_file)

    def update(self, agent_type: str, feedback: float, coherence: float):
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

    def get_agent_scores(self, query: str) -> Dict[str, float]:
        """Get current confidence scores for each agent"""
        return {agent: data['confidence']
                for agent, data in self.agent_weights.items()}
