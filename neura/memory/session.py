"""
Neura Memory Session
"""

from .message import Message


class Session:

    def __init__(self, user_id):

        self.user_id = user_id

        self.messages = []


    def add(self, role, content):

        self.messages.append(
            Message.create(role, content)
        )


    def history(self):

        return self.messages


    def clear(self):

        self.messages.clear()
