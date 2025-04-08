from .base_agent import GeneralAgent, AdmissionAgent, AIAgent
from .metrics import ChatbotMetrics
from .learning import ReinforcementLearner
from .utils import shared_ollama_model


class AgentCoordinator:
    def __init__(self):
        self.general_agent = GeneralAgent()
        self.admission_agent = AdmissionAgent()
        self.ai_agent = AIAgent()
        self.metrics = ChatbotMetrics()
        self.rl = ReinforcementLearner()
        self.last_interaction = None
        # Initialize LLM to avoid cold start
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
