"""
Neura Token
"""

from dataclasses import dataclass


@dataclass(slots=True)
class Token:

    id: int

    text: str
