# View: Handles user interaction
class View:
    def Display(self, message):
        print(message)

    def get_user_input(self, prompt):
        return input(prompt)
