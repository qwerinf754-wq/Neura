"""
Neura Chat Session
"""

class ChatSession:

    def __init__(self):
        self.messages = []

    def add_system(self, text):
        self.messages.append({
            "role": "system",
            "content": text
        })

    def add_user(self, text):
        self.messages.append({
            "role": "user",
            "content": text
        })

    def add_assistant(self, text):
        self.messages.append({
            "role": "assistant",
            "content": text
        })

    def history(self):
        return list(self.messages)

    def clear(self):
        self.messages.clear()

    def size(self):
        return len(self.messages)
