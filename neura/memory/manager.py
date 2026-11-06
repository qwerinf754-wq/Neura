"""
Neura Memory Manager
"""

from neura.logger import logger

from .session import Session


class MemoryManager:

    def __init__(self):

        self.sessions = {}


    def session(self, user_id):

        if user_id not in self.sessions:

            logger.info(
                f"Creating memory session: {user_id}"
            )

            self.sessions[user_id] = Session(user_id)

        return self.sessions[user_id]


    def count(self):

        return len(self.sessions)


MEMORY = MemoryManager()
