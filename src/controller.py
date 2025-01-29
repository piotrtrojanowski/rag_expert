# Controller: Handles application logic
class Controller:
    def __init__(self, model, view, event_manager):
        self.model = model
        self.view = view
        self.event_manager = event_manager

        # Subscribe to relevant events
        self.event_manager.subscribe("new_user_query", self.onNewUserQuery)

    def onNewUserQuery(self, user_query):
        response = self.model.generate_response(user_query)
        self.view.Display(response)
