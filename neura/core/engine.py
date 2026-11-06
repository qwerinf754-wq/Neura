"""
Neura Core Engine
"""

from neura.logger import logger

from .registry import REGISTRY


class CoreEngine:

    def __init__(self):

        self.running = False


    def startup(self):

        logger.info("Starting Core Engine")

        for module in REGISTRY.all():

            logger.info(f"Loading {module.name}")

            module.startup()

        self.running = True

        logger.info("Core Engine Ready")


    def shutdown(self):

        logger.info("Stopping Core Engine")

        for module in reversed(REGISTRY.all()):

            module.shutdown()

        self.running = False

        logger.info("Core Engine Stopped")


ENGINE = CoreEngine()
