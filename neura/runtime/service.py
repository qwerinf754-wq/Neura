"""
Neura Runtime Service
"""

class Service:

    name = "service"

    running = False

    def start(self):

        self.running = True

        return True


    def stop(self):

        self.running = False

        return True


    def status(self):

        return self.running
