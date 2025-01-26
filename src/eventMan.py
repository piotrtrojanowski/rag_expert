# Event Manager: Implements Pub-Sub mechanism
class EventManager:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, event_name, callback):
        if event_name not in self.subscribers:
            self.subscribers[event_name] = []
        self.subscribers[event_name].append(callback)

    def publish(self, event_name, *args, **kwargs):
        if event_name in self.subscribers:
            for callback in self.subscribers[event_name]:
                callback(*args, **kwargs)