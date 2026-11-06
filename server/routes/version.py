"""
Neura Version API
"""

from neura.version import VERSION


def version_payload():
    return {
        "status": "ok",
        "service": "neura-server",
        "version": VERSION.version,
        "stage": VERSION.stage,
        "codename": VERSION.codename,
    }
