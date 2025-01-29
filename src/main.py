from eventMan import EventManager
from model import Model
from view import View
from controller import Controller
import logging

def setup_logging():
    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('rag_expert.log')
    
    # Set formatter for handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Set levels
    console_handler.setLevel(logging.ERROR)
    file_handler.setLevel(logging.DEBUG)    # More detailed in file
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

def main():
    try:
      event_manager = EventManager()
      model = Model()
      view = View()
      controller = Controller(model, view, event_manager)
      view.Display("Welcome to the RAG Expert Project!")

      while True:
        user_input = view.get_user_input("Enter your query (or type 'quit' to exit): ")
        if user_input.lower() == "quit":  # Check if user wants to quit
            view.Display("Goodbye!")
            break

        # Publish the "update_message" event
        event_manager.publish("new_user_query", user_input)

    except Exception as e:
        print(f"Error in the program: {e}")

if __name__ == "__main__":
    setup_logging()
    main()
