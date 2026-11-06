"""
Neura Runtime Manager
"""

from neura.logger import logger

from .service import Service


class RuntimeManager:

    def __init__(self):

        self.services = {}


    def register(self, service: Service):

        self.services[service.name] = service

        logger.info(f"Registered service: {service.name}")


    def start(self):

        logger.info("Starting Runtime")

        for service in self.services.values():

            service.start()

            logger.info(f"Started {service.name}")

        logger.info("Runtime Ready")


    def stop(self):

        logger.info("Stopping Runtime")

        for service in reversed(list(self.services.values())):

            service.stop()

            logger.info(f"Stopped {service.name}")

        logger.info("Runtime Stopped")


    def status(self):

        return {
            name: service.status()
            for name, service in self.services.items()
        }


RUNTIME = RuntimeManager()
