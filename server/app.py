from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from neura.memory.auth_store import AuthStore, AuthStoreError
from neura.security.shield import validate_and_sanitize_prompt
from server.services.chat import ChatService, ChatServiceError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

auth_store = AuthStore(DATA_DIR / "neura.db")

HOST = "127.0.0.1"
DEFAULT_PORT = 8000

MAX_BODY_SIZE = 64 * 1024
MAX_PROMPT_LENGTH = 8000

ALLOWED_ORIGIN = "http://127.0.0.1:8000"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("neura.server")


# ============================================================
# CHAT SERVICE
# ============================================================

def create_chat_service(
    user_id: str,
) -> ChatService:
    """
    Construct ChatService using the trusted server-side identity.
    """

    return ChatService(
        user_id=user_id,
    )


# ============================================================
# HTTP HANDLER
# ============================================================

class NeuraServerHandler(BaseHTTPRequestHandler):

    server_version = "NeuraServer/2.0"

    def _set_common_headers(self) -> None:
        self.send_header(
            "Access-Control-Allow-Origin",
            ALLOWED_ORIGIN,
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization",
        )

        self.send_header(
            "X-Content-Type-Options",
            "nosniff",
        )

        self.send_header(
            "Cache-Control",
            "no-store",
        )

    # --------------------------------------------------------
    # OPTIONS
    # --------------------------------------------------------

    def do_OPTIONS(self) -> None:
        self.send_response(204)

        self._set_common_headers()

        self.end_headers()

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    def do_GET(self) -> None:

        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == "/health" or path == "/api/health":
            self._send_json_response(
                200,
                {
                    "status": "ok",
                    "service": "NeuraServer",
                },
            )
            return

        if path == "/api/auth/session":
            user = self._get_authenticated_user()

            if user is None:
                self._send_json_response(
                    401,
                    {
                        "status": "error",
                        "error": "unauthorized",
                    },
                )
                return

            self._send_json_response(
                200,
                {
                    "status": "ok",
                    "authenticated": True,
                    "user": {
                        "id": user["user_id"],
                        "username": user["username"],
                        "display_name": user["display_name"],
                    },
                },
            )
            return

        if path == "/" or path == "/index.html":
            frontend = PROJECT_ROOT / "Nova-Website" / "index.html"

            if not frontend.is_file():
                self._send_json_response(
                    500,
                    {
                        "status": "error",
                        "error": "Frontend not found.",
                    },
                )
                return

            payload = frontend.read_bytes()

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8",
            )
            self.send_header(
                "Content-Length",
                str(len(payload)),
            )
            self._set_common_headers()
            self.end_headers()

            try:
                self.wfile.write(payload)
            except BrokenPipeError:
                logger.debug(
                    "Client disconnected before frontend response was sent."
                )

            return

        self._send_json_response(
            404,
            {
                "error": "Endpoint not found",
            },
        )

    # --------------------------------------------------------
    # AUTH HELPERS
    # --------------------------------------------------------

    def _get_session_token(self) -> str | None:
        authorization = self.headers.get(
            "Authorization",
            "",
        ).strip()

        if not authorization:
            return None

        scheme, separator, token = authorization.partition(" ")

        if not separator:
            return None

        if scheme.lower() != "bearer":
            return None

        token = token.strip()

        return token or None

    def _get_authenticated_user(self) -> dict | None:
        token = self._get_session_token()

        if not token:
            return None

        try:
            return auth_store.validate_session(token)
        except Exception:
            logger.exception(
                "Session validation failed."
            )
            return None

    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    def do_POST(self) -> None:

        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # ----------------------------------------------------
        # REGISTER
        # ----------------------------------------------------

        if path == "/api/auth/register":
            try:
                body = self._read_request_body()
                data = json.loads(body.decode("utf-8"))

                if not isinstance(data, dict):
                    self._send_json_response(
                        400,
                        {
                            "status": "error",
                            "error": "Request body must be a JSON object.",
                        },
                    )
                    return

                username = data.get("username")
                password = data.get("password")
                display_name = data.get("display_name")

                if not isinstance(username, str):
                    self._send_json_response(
                        400,
                        {
                            "status": "error",
                            "error": "Username is required.",
                        },
                    )
                    return

                if not isinstance(password, str):
                    self._send_json_response(
                        400,
                        {
                            "status": "error",
                            "error": "Password is required.",
                        },
                    )
                    return

                if display_name is not None and not isinstance(
                    display_name,
                    str,
                ):
                    self._send_json_response(
                        400,
                        {
                            "status": "error",
                            "error": "Display name must be a string.",
                        },
                    )
                    return

                username = username.strip()

                if not username:
                    self._send_json_response(
                        400,
                        {
                            "status": "error",
                            "error": "Username cannot be empty.",
                        },
                    )
                    return

                if not password:
                    self._send_json_response(
                        400,
                        {
                            "status": "error",
                            "error": "Password cannot be empty.",
                        },
                    )
                    return

                user_id = auth_store.create_user(
                    username=username,
                    password=password,
                    display_name=(display_name or username).strip(),
                )

                user = auth_store.get_user_by_username(username)

                if user is None:
                    raise AuthStoreError(
                        "User was created but could not be loaded."
                    )

                self._send_json_response(
                    201,
                    {
                        "status": "ok",
                        "user": {
                            "id": user_id,
                            "username": user["username"],
                            "display_name": user["display_name"],
                        },
                    },
                )

            except AuthStoreError as exc:
                self._send_json_response(
                    409,
                    {
                        "status": "error",
                        "error": str(exc),
                    },
                )

            except UnicodeDecodeError:
                self._send_json_response(
                    400,
                    {
                        "status": "error",
                        "error": "Request body must be valid UTF-8.",
                    },
                )

            except json.JSONDecodeError:
                self._send_json_response(
                    400,
                    {
                        "status": "error",
                        "error": "Invalid JSON.",
                    },
                )

            except Exception:
                logger.exception(
                    "Unhandled error during registration."
                )
                self._send_json_response(
                    500,
                    {
                        "status": "error",
                        "error": "Internal server error.",
                    },
                )

            return

        # ----------------------------------------------------
        # LOGIN
        # ----------------------------------------------------

        if path == "/api/auth/login":
            try:
                body = self._read_request_body()
                data = json.loads(body.decode("utf-8"))

                if not isinstance(data, dict):
                    self._send_json_response(
                        400,
                        {
                            "status": "error",
                            "error": "Request body must be a JSON object.",
                        },
                    )
                    return

                username = data.get("username")
                password = data.get("password")

                if not isinstance(username, str) or not isinstance(
                    password,
                    str,
                ):
                    self._send_json_response(
                        400,
                        {
                            "status": "error",
                            "error": "Username and password are required.",
                        },
                    )
                    return

                user = auth_store.verify_password(
                    username.strip(),
                    password,
                )

                if user is None:
                    self._send_json_response(
                        401,
                        {
                            "status": "error",
                            "error": "Invalid username or password.",
                        },
                    )
                    return

                session_token, session_id = auth_store.create_session(
                    user_id=user["id"],
                )

                self._send_json_response(
                    200,
                    {
                        "status": "ok",
                        "session_token": session_token,
                        "session_id": session_id,
                        "user": {
                            "id": user["id"],
                            "username": user["username"],
                            "display_name": user["display_name"],
                        },
                    },
                )

            except UnicodeDecodeError:
                self._send_json_response(
                    400,
                    {
                        "status": "error",
                        "error": "Request body must be valid UTF-8.",
                    },
                )

            except json.JSONDecodeError:
                self._send_json_response(
                    400,
                    {
                        "status": "error",
                        "error": "Invalid JSON.",
                    },
                )

            except Exception:
                logger.exception(
                    "Unhandled error during login."
                )
                self._send_json_response(
                    500,
                    {
                        "status": "error",
                        "error": "Internal server error.",
                    },
                )

            return

        # ----------------------------------------------------
        # LOGOUT
        # ----------------------------------------------------

        if path == "/api/auth/logout":
            token = self._get_session_token()

            if token:
                try:
                    auth_store.revoke_session(token)
                except Exception:
                    logger.exception(
                        "Failed to revoke session."
                    )

            self._send_json_response(
                200,
                {
                    "status": "ok",
                },
            )
            return

        # ----------------------------------------------------
        # CHAT
        # ----------------------------------------------------

        if path != "/api/chat":
            self._send_json_response(
                404,
                {
                    "error": "Endpoint not found",
                },
            )
            return

        try:
            body = self._read_request_body()

            data = json.loads(
                body.decode("utf-8")
            )

            if not isinstance(data, dict):
                self._send_json_response(
                    400,
                    {
                        "error": "Request body must be a JSON object.",
                    },
                )
                return

            raw_prompt = data.get("prompt")

            if raw_prompt is None:
                raw_prompt = data.get("message")

            if not isinstance(raw_prompt, str):
                self._send_json_response(
                    400,
                    {
                        "error": "Field 'prompt' must be a string.",
                    },
                )
                return

            if len(raw_prompt) > MAX_PROMPT_LENGTH:
                self._send_json_response(
                    400,
                    {
                        "error": "Prompt is too long.",
                    },
                )
                return

            safe_prompt = validate_and_sanitize_prompt(
                raw_prompt
            )

            if not safe_prompt:
                self._send_json_response(
                    400,
                    {
                        "error": "Prompt cannot be empty.",
                    },
                )
                return

            # ------------------------------------------------
            # AUTHENTICATION
            # ------------------------------------------------

            user = self._get_authenticated_user()

            if user is None:
                self._send_json_response(
                    401,
                    {
                        "error": "Authentication required.",
                    },
                )
                return

            user_id = user["user_id"]

            # ------------------------------------------------
            # CHAT PAYLOAD
            # ------------------------------------------------

            chat_payload = dict(data)

            chat_payload["message"] = safe_prompt

            chat_payload.pop(
                "user_id",
                None,
            )

            # ------------------------------------------------
            # CHAT SERVICE
            # ------------------------------------------------

            chat_service = create_chat_service(
                user_id=user_id,
            )

            result = chat_service.handle_chat(
                chat_payload
            )

            if not isinstance(result, dict):
                self._send_json_response(
                    500,
                    {
                        "error": "Invalid ChatService response.",
                    },
                )
                return

            if result.get("status") == "error":
                self._send_json_response(
                    400,
                    result,
                )
                return

            self._send_json_response(
                200,
                result,
            )

        except UnicodeDecodeError:
            self._send_json_response(
                400,
                {
                    "error": "Request body must be valid UTF-8.",
                },
            )

        except json.JSONDecodeError:
            self._send_json_response(
                400,
                {
                    "error": "Invalid JSON.",
                },
            )

        except ChatServiceError as exc:
            logger.warning(
                "ChatService error: %s",
                exc,
            )
            self._send_json_response(
                400,
                {
                    "error": str(exc),
                },
            )

        except ValueError as exc:
            self._send_json_response(
                400,
                {
                    "error": str(exc),
                },
            )

        except Exception:
            logger.exception(
                "Unhandled error while processing request."
            )

            if not self.wfile.closed:
                self._send_json_response(
                    500,
                    {
                        "error": "Internal server error.",
                    },
                )

    # --------------------------------------------------------
    # REQUEST BODY
    # --------------------------------------------------------

    def _read_request_body(self) -> bytes:

        raw_length = self.headers.get(
            "Content-Length"
        )

        if raw_length is None:
            raise ValueError(
                "Content-Length header is required."
            )

        try:
            content_length = int(
                raw_length
            )

        except (
            TypeError,
            ValueError,
        ):
            raise ValueError(
                "Invalid Content-Length."
            )

        if content_length < 0:
            raise ValueError(
                "Invalid Content-Length."
            )

        if content_length > MAX_BODY_SIZE:
            raise ValueError(
                "Request body is too large."
            )

        return self.rfile.read(
            content_length
        )

    # --------------------------------------------------------
    # JSON RESPONSE
    # --------------------------------------------------------

    def _send_json_response(
        self,
        code: int,
        data: dict,
    ) -> None:

        payload = json.dumps(
            data,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(
            code
        )

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(payload)),
        )

        self._set_common_headers()

        self.end_headers()

        try:
            self.wfile.write(
                payload
            )

        except BrokenPipeError:
            logger.debug(
                "Client disconnected before response was sent."
            )

    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    def log_message(
        self,
        format: str,
        *args,
    ) -> None:

        logger.info(
            "%s - %s",
            self.address_string(),
            format % args,
        )


# ============================================================
# SERVER
# ============================================================

def run_server(
    port: int = DEFAULT_PORT,
) -> None:

    if not isinstance(
        port,
        int,
    ):
        raise TypeError(
            "port must be an integer."
        )

    if not 1 <= port <= 65535:
        raise ValueError(
            "port must be between 1 and 65535."
        )

    server_address = (
        HOST,
        port,
    )

    httpd = ThreadingHTTPServer(
        server_address,
        NeuraServerHandler,
    )

    logger.info(
        "Neura server running on http://%s:%d",
        HOST,
        port,
    )

    try:
        httpd.serve_forever()

    except KeyboardInterrupt:
        logger.info(
            "Neura server shutting down."
        )

    finally:
        httpd.server_close()


if __name__ == "__main__":
    run_server()
