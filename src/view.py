# View: Handles user interaction
class View:
    def Display(self, message):
        print(f"Message: {message}")

    def get_user_input(self, prompt):
        return input(prompt)
