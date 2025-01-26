# Model: Manages data
from ragExpert import RAGExpert

class Model:
    def __init__(self):
        self.ragExpert = RAGExpert()

    def GenerateResponse(self, user_query):
        return self.ragExpert.Respond(user_query)

