from .coordinator import AgentCoordinator


def run_cli():
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
                feedback = int(
                    input("\nPlease rate this response (1-5, where 5 is best): "))
                if 1 <= feedback <= 5:
                    coordinator.add_user_feedback(feedback)
                    break
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")


if __name__ == "__main__":
    run_cli()
