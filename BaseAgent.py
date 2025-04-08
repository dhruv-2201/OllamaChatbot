"""
BaseAgent module (compatibility layer)
This file provides backward compatibility for existing imports.
For new code, import directly from the agents package.
"""

# Import all components from the new modular structure
from agents.utils import shared_embedding_model, shared_ollama_model
from agents.memory import FAISSManager, SimpleConversationMemory
from agents.base_agent import BaseAgent, GeneralAgent, AdmissionAgent, AIAgent
from agents.coordinator import AgentCoordinator
from agents.metrics import ChatbotMetrics
from agents.learning import ReinforcementLearner

# Enable CLI functionality when running this file directly
if __name__ == "__main__":
    from agents.cli import run_cli
    run_cli()
