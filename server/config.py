"""
Neura Server Configuration
"""

import os


class ServerConfig:
    """
    Central configuration for the Neura HTTP server.
    """

    def __init__(
        self,
        host=None,
        port=None,
        debug=None,
    ):
        self.host = (
            host
            if host is not None
            else os.getenv(
                "NEURA_SERVER_HOST",
                "127.0.0.1",
            )
        )

        self.port = (
            int(port)
            if port is not None
            else int(
                os.getenv(
                    "NEURA_SERVER_PORT",
                    "8000",
                )
            )
        )

        if debug is None:
            self.debug = (
                os.getenv(
                    "NEURA_SERVER_DEBUG",
                    "0",
                )
                == "1"
            )
        else:
            self.debug = bool(debug)

        # -------------------------------------------------
        # Closed Beta Authentication
        # -------------------------------------------------

        self.api_key = os.getenv(
            "NEURA_BETA_API_KEY",
            "",
        ).strip()

        # -------------------------------------------------
        # Request Limits
        # -------------------------------------------------

        self.max_request_bytes = int(
            os.getenv(
                "NEURA_MAX_REQUEST_BYTES",
                "65536",
            )
        )

        self.max_message_chars = int(
            os.getenv(
                "NEURA_MAX_MESSAGE_CHARS",
                "8192",
            )
        )

        self.max_tokens = int(
            os.getenv(
                "NEURA_MAX_TOKENS",
                "64",
            )
        )

        # -------------------------------------------------
        # CORS
        # -------------------------------------------------

        self.cors_origin = os.getenv(
            "NEURA_CORS_ORIGIN",
            "http://127.0.0.1:8080",
        ).strip()

    def as_dict(self):
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "authentication_enabled": bool(
                self.api_key
            ),
            "max_request_bytes": self.max_request_bytes,
            "max_message_chars": self.max_message_chars,
            "max_tokens": self.max_tokens,
            "cors_origin": self.cors_origin,
        }
