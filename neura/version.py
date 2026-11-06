"""
Neura 11.0
Version Information
"""

from dataclasses import dataclass


__version__ = "11.0.0"
__stage__ = "Foundation"
__codename__ = "Genesis"


@dataclass(slots=True)
class VersionInfo:

    name: str
    version: str
    stage: str
    codename: str


VERSION = VersionInfo(
    name="Neura",
    version=__version__,
    stage=__stage__,
    codename=__codename__,
)


def get_version():

    return VERSION
