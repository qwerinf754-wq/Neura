"""
Neura Runtime Status API
"""


def status_payload(runtime):
    if runtime is None:
        raise ValueError("runtime is required")

    status = runtime.status()

    if not isinstance(status, dict):
        raise TypeError(
            "runtime.status() must return a dictionary"
        )

    return {
        "status": "ok",
        "service": "neura-server",
        "runtime": status,
    }
