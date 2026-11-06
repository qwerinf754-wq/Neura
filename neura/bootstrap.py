"""
Neura Bootstrap
"""

from neura.version import VERSION
from neura.config import CONFIG
from neura.logger import logger

from neura.core import ENGINE
from neura.runtime import RUNTIME


def start():

    logger.info(f"{VERSION.name} {VERSION.version}")
    logger.info(f"Stage : {VERSION.stage}")
    logger.info(f"Codename : {VERSION.codename}")
    logger.info(f"Server : {CONFIG.host}:{CONFIG.port}")

    ENGINE.startup()

    RUNTIME.start()


if __name__ == "__main__":
    start()
