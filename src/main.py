from eventMan import EventManager
from model import Model
from view import View
from controller import Controller

def main():
    try:
      print("Welcome to the RAG Expert Project!")

      event_manager = EventManager()
      model = Model()
      view = View()
      controller = Controller(model, view, event_manager)

      while True:
        user_input = view.get_user_input("Enter your query (or type 'quit' to exit): ")
        if user_input.lower() == "quit":  # Check if user wants to quit
            print("Goodbye!")
            break

        # Publish the "update_message" event
        event_manager.publish("new_user_query", user_input)

    except Exception as e:
        print(f"Error in the program: {e}")

if __name__ == "__main__":
    main()
