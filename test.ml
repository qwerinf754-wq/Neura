#!/usr/bin/env python3
"""Neura attack simulation (defensive). Localhost only by default.

Usage:
  python3 neura_attack_test.py --port 8000 --max-body 1048576 --rate 60 --timeout 10
"""
import argparse
import socket
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--max-body", type=int, default=1048576, help="MAX_REQUEST_BODY")
ap.add_argument("--rate", type=int, default=60, help="RATE_MAX_REQUESTS")
ap.add_argument("--timeout", type=int, default=10, help="REQUEST_TIMEOUT_SECONDS")
ap.add_argument("--allow-remote", action="store_true")
a = ap.parse_args()

if a.host not in ("127.0.0.1", "localhost", "::1") and not a.allow_remote:
    sys.exit("Refusing non-local host. Only test servers you own (--allow-remote).")

results, bodies = [], []


def raw(payload: bytes, timeout=5) -> bytes:
    s = socket.create_connection((a.host, a.port), timeout=timeout)
    chunks, total = [], 0
    try:
        try:
            s.sendall(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
        while total < 2_000_000:
            try:
                d = s.recv(65536)
            except (socket.timeout, ConnectionResetError):
                break
            if not d:
                break
            chunks.append(d)
            total += len(d)
    finally:
        s.close()
    return b"".join(chunks)


def status(resp: bytes) -> int:
    try:
        return int(resp.split(b" ", 2)[1])
    except Exception:
        return 0


def req(method, path, body=b"", headers=None, cl=None) -> bytes:
    h = {"Host": f"{a.host}:{a.port}", "Connection": "close"}
    h.update(headers or {})
    if body or cl is not None or method == "POST":
        h["Content-Length"] = str(len(body) if cl is None else cl)
    head = f"{method} {path} HTTP/1.1\r\n"
    head += "".join(f"{k}: {v}\r\n" for k, v in h.items()) + "\r\n"
    return head.encode("latin-1") + body


def send(*args, **kw):
    r = raw(req(*args, **kw))
    bodies.append(r)
    return status(r), r


def check(name, ok, info=""):
    results.append(ok)
    print(("PASS" if ok else "FAIL"), "-", name, info)


# 1) Path traversal / sensitive files
TRAV = [
    "/../checkpoints/neura_chat_sft_best.pt",
    "/..%2Fcheckpoints%2Fneura_chat_sft_best.pt",
    "/%2e%2e/%2e%2e/checkpoints/neura_chat_sft_best.pt",
    "/%252e%252e%252fcheckpoints%252fneura_chat_sft_best.pt",
    "/foo/../checkpoints/neura_chat_sft_best.pt",
    "/..\\checkpoints\\neura_chat_sft_best.pt",
    "/%5c..%5ccheckpoints%5cneura_chat_sft_best.pt",
    "//etc/passwd",
    "/../../../../etc/passwd",
    "/%c0%ae%c0%ae/%c0%ae%c0%ae/etc/passwd",
    "/server/app.py",
    "/.env",
    "/.git/config",
    "/foo/%00/bar",
]
LEAK = [b"PK\x03\x04", b"root:x:", b"MAX_REQUEST_BODY", b"[core]"]
print("== Path traversal ==")
for p in TRAV:
    st, r = send("GET", p)
    leaked = any(m in r for m in LEAK)
    check(f"GET {p}", st != 200 and not leaked, f"(status={st}, leaked={leaked})")

# 2) Oversized body
print("== Oversized body ==")
st, _ = send("POST", "/api/chat", headers={"Content-Type": "application/json"}, cl=999_999_999)
check("declared huge Content-Length -> 413", st in (413, 400), f"(status={st})")
big = b"a" * (a.max_body + 1024)
st, _ = send("POST", "/api/chat", body=big, headers={"Content-Type": "application/json"})
check("actual oversized body -> 413", st in (413, 400), f"(status={st})")

# 3) Malformed input
print("== Malformed input ==")
st, r = send("POST", "/api/chat", body=b"{bad json", headers={"Content-Type": "application/json"})
check("bad JSON -> 4xx", 400 <= st < 500, f"(status={st})")
st, r = send("POST", "/api/chat", body=b"[]", headers={"Content-Type": "application/json"})
check("JSON wrong shape -> 4xx", 400 <= st < 500, f"(status={st})")
st, r = send("POST", "/api/chat", body=b'{"message": ' + b'"x"' * 10 + b"}",
             headers={"Content-Type": "application/json"})
check("garbage fields -> not 5xx", st < 500, f"(status={st})")

# 4) Unexpected methods
print("== Methods ==")
for m in ("PUT", "DELETE", "TRACE", "PATCH"):
    st, _ = send(m, "/api/chat")
    check(f"{m} /api/chat rejected", st in (0, 400, 404, 405, 501), f"(status={st})")

# 5) Header abuse
print("== Header abuse ==")
st, _ = send("GET", "/", headers={"X-Junk": "a" * 70000})
check("70KB header rejected", st in (0, 400, 413, 431), f"(status={st})")
st, r = send("GET", "/%0d%0aSet-Cookie:x=1")
check("CRLF injection not reflected", b"\r\nSet-Cookie: x=1" not in r, f"(status={st})")

# 6) Slow request (slowloris-lite)
print("== Slow request ==")
s = socket.create_connection((a.host, a.port), timeout=5)
s.sendall(f"GET / HTTP/1.1\r\nHost: {a.host}\r\n".encode())
s.settimeout(1)
t0, closed = time.time(), False
while time.time() - t0 < a.timeout + 4:
    try:
        d = s.recv(4096)
        if not d or d.startswith(b"HTTP/1.") :
            closed = True
            break
    except socket.timeout:
        continue
    except OSError:
        closed = True
        break
s.close()
check(f"incomplete request dropped within ~{a.timeout}s", closed, f"(after {time.time()-t0:.1f}s)")

# 7) Rate limit (last: it may lock you out for a while)
print("== Rate limit ==")
codes = [send("GET", "/api/health")[0] for _ in range(a.rate * 2 + 10)]
check("429 appears after the limit", 429 in codes, f"(429 count={codes.count(429)})")

# 8) Error leakage
print("== Error leakage ==")
leak = any(b"Traceback" in r or b"/home/" in r or b"File \"" in r for r in bodies)
check("no stack traces / paths in responses", not leak)

print()
failed = results.count(False)
print(f"RESULT: {len(results)-failed}/{len(results)} passed")
sys.exit(1 if failed else 0)

