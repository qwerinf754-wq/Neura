"""
Neura Capabilities API
"""


def capabilities_payload(runtime):
    if runtime is None:
        raise ValueError(
            "runtime is required"
        )

    status = runtime.status()

    registry = status.get(
        "registry",
        {},
    )

    return {
        "status": "ok",
        "service": "neura-server",

        "capabilities": {
            "runtime": True,
            "plugins": True,
            "extensions": True,

            "tools": (
                registry.get("tools", 0) > 0
            ),

            "agents": (
                registry.get("agents", 0) > 0
            ),

            "models": (
                registry.get("models", 0) > 0
            ),

            "memory": (
                registry.get("memories", 0) > 0
            ),

            "rag": (
                registry.get("rag", 0) > 0
            ),
        },

        "counts": registry,
    }
