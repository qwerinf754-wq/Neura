"""
Neura Chat API
"""

class ChatRouteError(Exception):
    """Raised when a chat request is invalid."""


def chat_payload(
    message,
    max_tokens=32,
    service=None,
):
    if not isinstance(message, str):
        raise ChatRouteError(
            "message must be a string"
        )

    message = message.strip()

    if not message:
        raise ChatRouteError(
            "message cannot be empty"
        )

    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise ChatRouteError(
            "max_tokens must be an integer"
        )

    if max_tokens <= 0:
        raise ChatRouteError(
            "max_tokens must be positive"
        )

    if service is None:
        raise ChatRouteError(
            "chat service is required"
        )

    response = service.chat(
        message,
        max_tokens=max_tokens,
    )

    return {
        "status": "ok",
        "service": "neura-server",
        "message": message,
        "response": response,
    }
