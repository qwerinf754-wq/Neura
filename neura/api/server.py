from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
import requests

from neura.inference.sft_runtime import SFTRuntime

HOST = "127.0.0.1"
PORT = 8000

CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2]
    / "checkpoints"
    / "neura_chat_sft_best.pt"
)

DEFAULT_MAX_TOKENS = 64
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REPETITION_PENALTY = 1.18
DEFAULT_NO_REPEAT_NGRAM_SIZE = 2
DEFAULT_REPEAT_TOKEN_LIMIT = 3

_RUNTIME: SFTRuntime | None = None
_RUNTIME_LOCK = threading.Lock()

def get_runtime() -> SFTRuntime:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            try:
                _RUNTIME = SFTRuntime(
                    checkpoint_path=CHECKPOINT_PATH,
                    device="cpu",
                )
            except Exception:
                _RUNTIME = None
    return _RUNTIME

def generate_local_super_response(message: str) -> str | None:
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "أنت Neura، نظام ذكاء اصطناعي محلي خارق ومتطور جداً ومبرمج محترف. "
                            "قدم إجابات دقيقة، عميقة، ومباشرة، مع تنسيق الأكواد البرمجية والجداول باحترافية."
                        ),
                    },
                    {"role": "user", "content": message},
                ],
                "stream": False,
            },
            timeout=15,
        )
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "").strip()
    except Exception:
        pass
    return None

def _number(data: dict[str, Any], key: str, default: Any) -> Any:
    value = data.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return value

class NeuraRequestHandler(BaseHTTPRequestHandler):
    server_version = "NeuraServer/0.1.0"

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_json({
                "name": "Neura",
                "server": "Neura HTTP Server",
                "version": "0.1.0",
                "status": "running",
                "generation": "hybrid-supercharged",
                "checkpoint": CHECKPOINT_PATH.name,
            })
            return

        if self.path == "/health":
            runtime = get_runtime()
            runtime_loaded = runtime is not None and getattr(runtime, "loaded", True)
            self._send_json({
                "status": "ok",
                "service": "neura",
                "version": "0.1.0",
                "runtime": "hybrid-supercharged",
                "runtime_loaded": runtime_loaded,
            })
            return

        self._send_json({"status": "error", "error": "Not Found", "path": self.path}, status=404)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("Empty request body")
        raw = self.rfile.read(length)
        if not raw:
            raise ValueError("Empty request body")
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _handle_chat(self) -> None:
        try:
            data = self._read_json_body()
            message = data.get("message") or data.get("prompt")

            if not isinstance(message, str) or not message.strip():
                self._send_json({"status": "error", "error": "message cannot be empty"}, status=400)
                return

            message = message.strip()
            conversation_id = data.get("conversation_id") or "local-neura-session"

            response_text = generate_local_super_response(message)
            runtime_used = "ollama-local-super"

            if not response_text:
                runtime_used = "sft-fallback"
                max_tokens = _number(data, "max_tokens", DEFAULT_MAX_TOKENS)
                temperature = _number(data, "temperature", DEFAULT_TEMPERATURE)
                repetition_penalty = _number(data, "repetition_penalty", DEFAULT_REPETITION_PENALTY)
                no_repeat_ngram_size = _number(data, "no_repeat_ngram_size", DEFAULT_NO_REPEAT_NGRAM_SIZE)
                repeat_token_limit = _number(data, "repeat_token_limit", DEFAULT_REPEAT_TOKEN_LIMIT)

                runtime = get_runtime()
                if runtime is not None:
                    result = runtime.generate(
                        message,
                        max_new_tokens=int(max_tokens),
                        temperature=float(temperature),
                        repetition_penalty=float(repetition_penalty),
                        no_repeat_ngram_size=int(no_repeat_ngram_size),
                        repeat_token_limit=int(repeat_token_limit),
                    )
                    response_text = result.text.strip()
                    finish_reason = getattr(result, "finish_reason", "stop")
                    prompt_tokens_len = len(getattr(result, "prompt_tokens", []))
                    generated_tokens_len = len(getattr(result, "generated_tokens", []))
                else:
                    response_text = "عذراً، تعذر العثور على نموذج التشغيل المحلي."
                    finish_reason = "error"
                    prompt_tokens_len = 0
                    generated_tokens_len = 0
            else:
                finish_reason = "stop"
                prompt_tokens_len = len(message.split())
                generated_tokens_len = len(response_text.split())

            self._send_json({
                "status": "ok",
                "model": "Neura-Supercharged",
                "runtime": runtime_used,
                "conversation_id": conversation_id,
                "message": message,
                "response": response_text,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens_len,
                "generated_tokens": generated_tokens_len,
            })

        except json.JSONDecodeError:
            self._send_json({"status": "error", "error": "Invalid JSON"}, status=400)
        except ValueError as exc:
            self._send_json({"status": "error", "error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"status": "error", "error": "Internal Server Error", "detail": str(exc)}, status=500)

    def do_POST(self) -> None:
        if self.path in ["/api/chat", "/generate"]:
            self._handle_chat()
            return
        self._send_json({"status": "error", "error": "Not Found", "path": self.path}, status=404)

    def log_message(self, format: str, *args: Any) -> None:
        print("[Neura Server]", format % args)

def create_server() -> ThreadingHTTPServer:
    return ThreadingHTTPServer((HOST, PORT), NeuraRequestHandler)

def main() -> None:
    server = create_server()
    print("=" * 64)
    print("Neura Supercharged HTTP Server")
    print("=" * 64)
    print(f"Address : http://{HOST}:{PORT}")
    print("Health  : /health")
    print("Chat    : POST /api/chat")
    print("Runtime : Hybrid (Local Super + SFT Fallback)")
    print("Status  : RUNNING")
    print("=" * 64)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Neura server...")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
