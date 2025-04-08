from langchain.prompts import PromptTemplate
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from .utils import shared_ollama_model
from .memory import SimpleConversationMemory, FAISSManager


class BaseAgent:
    def __init__(self):
        self.llm = shared_ollama_model
        self.memory = SimpleConversationMemory()
        self.wikipedia_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper())
        self.use_memory = False
        self.faiss_db = None

    def is_factual_query(self, user_query):
        """Determine if a query is factual (basic heuristic)."""
        factual_keywords = ["what", "who", "when", "where",
                            "capital", "history", "define", "explain"]
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
