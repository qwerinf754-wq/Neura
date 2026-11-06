"""
Neura Defensive Security Shield
================================

Defensive security boundary for untrusted text and request metadata.

Design goals:
- prompt-injection detection
- Unicode / hidden-character normalization
- command/code injection detection
- path traversal detection
- XSS / template injection detection
- SSRF-like local-target detection
- credential/secret harvesting detection
- bounded input processing
- sliding-window rate limiting
- temporary source blocking
- append-only audit logging without raw prompt storage
- isolated honeypot recording

This module is defensive only.

It NEVER:
- executes attacker supplied code
- evaluates attacker supplied expressions
- opens arbitrary network connections
- scans remote systems
- reads arbitrary files
- attempts exploitation
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import threading
import time
import unicodedata
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

# Hardened defensive deception layers.
from neura.security.honeypot import (
    IsolatedHoneypot as HardenedIsolatedHoneypot,
)
from neura.security.spider_web import (
    SpiderWeb,
    attach_to_shield as attach_spider_web_to_shield,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_PROMPT_LENGTH = 8000
MAX_SOURCE_LENGTH = 256

RATE_LIMIT_WINDOW = 60.0
RATE_LIMIT_MAX_REQUESTS = 30

DEFAULT_BLOCK_SECONDS = 300.0

AUDIT_LOG_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "security"
    / "audit.jsonl"
)

CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
)

HIDDEN_UNICODE_RE = re.compile(
    r"[\u200B-\u200F\u202A-\u202E\u2060-\u2064\u2066-\u206F\uFEFF]"
)

MULTISPACE_RE = re.compile(r"\s+")


# ============================================================
# SECURITY EVENT
# ============================================================

@dataclass(frozen=True)
class SecurityEvent:
    timestamp: float
    event: str
    source: str
    severity: str
    details: dict[str, Any]


# ============================================================
# AUDIT LOGGER
# ============================================================

class SecurityAuditLogger:
    """Thread-safe append-only JSONL audit logger."""

    def __init__(
        self,
        path: Path | str = AUDIT_LOG_PATH,
    ) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(
        self,
        event: str,
        source: str = "unknown",
        severity: str = "info",
        **details: Any,
    ) -> None:
        entry = SecurityEvent(
            timestamp=time.time(),
            event=str(event),
            source=_source_key(source),
            severity=str(severity),
            details=dict(details),
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = json.dumps(
            asdict(entry),
            ensure_ascii=False,
            separators=(",", ":"),
        )

        with self._lock:
            with self.path.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(payload + "\n")


AUDIT_LOGGER = SecurityAuditLogger()


# ============================================================
# SOURCE KEY
# ============================================================

def _source_key(source: str) -> str:
    """
    Return a stable non-reversible source identifier.

    Raw source identifiers are never written to the audit log.
    """
    value = str(source or "unknown").strip()

    if len(value) > MAX_SOURCE_LENGTH:
        value = value[:MAX_SOURCE_LENGTH]

    return hashlib.sha256(
        value.encode("utf-8", errors="replace")
    ).hexdigest()[:32]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def _normalize_text(value: str) -> str:
    """
    Normalize untrusted text without executing or interpreting it.

    NFKC reduces common Unicode confusables while hidden formatting
    characters and control characters are removed.
    """
    if not isinstance(value, str):
        return ""

    value = value[:MAX_PROMPT_LENGTH]

    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    value = HIDDEN_UNICODE_RE.sub(
        "",
        value,
    )

    value = CONTROL_CHARS_RE.sub(
        "",
        value,
    )

    value = MULTISPACE_RE.sub(
        " ",
        value,
    )

    return value.strip()


# ============================================================
# RATE LIMITER
# ============================================================

class RateLimiter:
    """Thread-safe sliding-window rate limiter."""

    def __init__(
        self,
        max_requests: int = RATE_LIMIT_MAX_REQUESTS,
        window_seconds: float = RATE_LIMIT_WINDOW,
    ) -> None:
        if max_requests <= 0:
            raise ValueError(
                "max_requests must be positive"
            )

        if window_seconds <= 0:
            raise ValueError(
                "window_seconds must be positive"
            )

        self.max_requests = int(max_requests)
        self.window_seconds = float(window_seconds)

        self._requests: defaultdict[str, deque[float]] = (
            defaultdict(deque)
        )

        self._lock = threading.Lock()

    def _purge(
        self,
        timestamps: deque[float],
        now: float,
    ) -> None:
        cutoff = now - self.window_seconds

        while (
            timestamps
            and timestamps[0] <= cutoff
        ):
            timestamps.popleft()

    def allow(
        self,
        source: str,
    ) -> bool:
        key = _source_key(source)
        now = time.monotonic()

        with self._lock:
            timestamps = self._requests[key]

            self._purge(
                timestamps,
                now,
            )

            if len(timestamps) >= self.max_requests:
                return False

            timestamps.append(now)
            return True

    def remaining(
        self,
        source: str,
    ) -> int:
        key = _source_key(source)
        now = time.monotonic()

        with self._lock:
            timestamps = self._requests[key]

            self._purge(
                timestamps,
                now,
            )

            return max(
                0,
                self.max_requests - len(timestamps),
            )

    def clear(
        self,
        source: str,
    ) -> None:
        key = _source_key(source)

        with self._lock:
            self._requests.pop(
                key,
                None,
            )

    def cleanup(self) -> None:
        """Remove expired and unused source buckets."""
        now = time.monotonic()

        with self._lock:
            stale: list[str] = []

            for key, timestamps in self._requests.items():
                self._purge(
                    timestamps,
                    now,
                )

                if not timestamps:
                    stale.append(key)

            for key in stale:
                self._requests.pop(
                    key,
                    None,
                )


# ============================================================
# TEMPORARY BLOCK LIST
# ============================================================

class TemporaryBlockList:
    """Thread-safe temporary source block list."""

    def __init__(
        self,
        default_seconds: float = DEFAULT_BLOCK_SECONDS,
    ) -> None:
        if default_seconds <= 0:
            raise ValueError(
                "default_seconds must be positive"
            )

        self.default_seconds = float(
            default_seconds
        )

        self._blocked: dict[str, float] = {}
        self._lock = threading.Lock()

    def block(
        self,
        source: str,
        seconds: float | None = None,
    ) -> None:
        duration = (
            self.default_seconds
            if seconds is None
            else float(seconds)
        )

        if duration <= 0:
            raise ValueError(
                "block duration must be positive"
            )

        key = _source_key(source)

        with self._lock:
            self._blocked[key] = (
                time.monotonic() + duration
            )

    def is_blocked(
        self,
        source: str,
    ) -> bool:
        key = _source_key(source)
        now = time.monotonic()

        with self._lock:
            expires = self._blocked.get(key)

            if expires is None:
                return False

            if expires <= now:
                self._blocked.pop(
                    key,
                    None,
                )
                return False

            return True

    def unblock(
        self,
        source: str,
    ) -> None:
        key = _source_key(source)

        with self._lock:
            self._blocked.pop(
                key,
                None,
            )

    def cleanup(self) -> None:
        """Remove expired blocks."""
        now = time.monotonic()

        with self._lock:
            expired = [
                key
                for key, expires in self._blocked.items()
                if expires <= now
            ]

            for key in expired:
                self._blocked.pop(
                    key,
                    None,
                )


# ============================================================
# THREAT DETECTOR
# ============================================================

class ThreatDetector:
    """
    Deterministic defensive detector.

    Detection is intentionally conservative: a match is a signal,
    not proof of malicious intent.
    """

    PATTERNS = (
        (
            "prompt_injection",
            re.compile(
                r"(?i)"
                r"(ignore\s+(all\s+)?previous\s+instructions|"
                r"ignore\s+the\s+system\s+prompt|"
                r"system\s+prompt\s+override|"
                r"override\s+system\s+instructions|"
                r"reveal\s+(the\s+)?system\s+prompt|"
                r"print\s+(the\s+)?system\s+prompt|"
                r"تجاهل\s+التعليمات\s+السابقة|"
                r"تجاهل\s+تعليمات\s+النظام|"
                r"تجاوز\s+تعليمات\s+النظام|"
                r"اكشف\s+تعليمات\s+النظام)"
            ),
            "high",
        ),
        (
            "code_execution_probe",
            re.compile(
                r"(?i)"
                r"(\bexec\s*\(|"
                r"\beval\s*\(|"
                r"__import__\s*\(|"
                r"subprocess\s*\.\s*"
                r"(run|Popen|call|check_output)|"
                r"os\s*\.\s*(system|popen|spawn|exec)|"
                r"shell\s*=\s*true|"
                r"child_process|"
                r"Runtime\.getRuntime\s*\(|"
                r"ProcessBuilder\s*\()"
            ),
            "high",
        ),
        (
            "path_traversal",
            re.compile(
                r"(?i)"
                r"(\.\./|\.\.\\|"
                r"%2e%2e%2f|"
                r"%2e%2e%5c|"
                r"%252e%252e%252f|"
                r"%252e%252e%255c)"
            ),
            "high",
        ),
        (
            "credential_probe",
            re.compile(
                r"(?i)"
                r"(password\s*dump|"
                r"credential\s*dump|"
                r"steal\s+(password|token|cookie|secret)|"
                r"dump\s+(passwords|tokens|cookies|credentials)|"
                r"extract\s+(passwords|tokens|cookies|credentials)|"
                r"سرقة\s+(كلمات\s+المرور|الرموز|ملفات\s+الارتباط)|"
                r"استخراج\s+(كلمات\s+المرور|الرموز|بيانات\s+الدخول))"
            ),
            "high",
        ),
        (
            "script_probe",
            re.compile(
                r"(?i)"
                r"(<script\b|"
                r"</script\s*>|"
                r"javascript\s*:|"
                r"vbscript\s*:|"
                r"data\s*:\s*text/html|"
                r"onerror\s*=|"
                r"onload\s*=|"
                r"onclick\s*=)"
            ),
            "medium",
        ),
        (
            "template_injection",
            re.compile(
                r"(?i)"
                r"(\{\{.*\}\}|"
                r"\{%.*%\}|"
                r"\$\{.*\}|"
                r"<%.*%>)"
            ),
            "medium",
        ),
        (
            "shell_injection",
            re.compile(
                r"(?i)"
                r"(^|[\s;&|])"
                r"(curl|wget|nc|ncat|bash|sh|zsh|powershell|cmd)"
                r"(\s|$|[;&|])"
            ),
            "high",
        ),
        (
            "secret_file_probe",
            re.compile(
                r"(?i)"
                r"(/etc/passwd|"
                r"/etc/shadow|"
                r"\.ssh/id_(rsa|ed25519|ecdsa)|"
                r"\.env\b|"
                r"authorized_keys|"
                r"wallet\.dat)"
            ),
            "high",
        ),
    )

    def inspect(
        self,
        text: str,
    ) -> dict[str, Any]:
        if not isinstance(text, str):
            return {
                "suspicious": True,
                "severity": "high",
                "matches": [
                    "non_string_input"
                ],
            }

        normalized = _normalize_text(text)

        if len(text) > MAX_PROMPT_LENGTH:
            return {
                "suspicious": True,
                "severity": "high",
                "matches": [
                    "input_too_large"
                ],
            }

        matches: list[str] = []
        severities: list[str] = []

        for (
            name,
            pattern,
            severity,
        ) in self.PATTERNS:
            if pattern.search(normalized):
                matches.append(name)
                severities.append(severity)

        url_result = self._inspect_urls(
            normalized
        )

        if url_result:
            matches.extend(url_result)
            severities.append("high")

        if "high" in severities:
            final_severity = "high"
        elif "medium" in severities:
            final_severity = "medium"
        else:
            final_severity = "low"

        return {
            "suspicious": bool(matches),
            "severity": final_severity,
            "matches": matches,
        }

    @staticmethod
    def _inspect_urls(
        text: str,
    ) -> list[str]:
        matches: list[str] = []

        urls = re.findall(
            r"https?://[^\s<>'\"]+",
            text,
            flags=re.IGNORECASE,
        )

        for raw_url in urls[:8]:
            try:
                decoded = unquote(raw_url)
                parsed = urlsplit(decoded)

                hostname = parsed.hostname

                if not hostname:
                    continue

                host_lower = hostname.lower()

                if host_lower in {
                    "localhost",
                    "localhost.localdomain",
                }:
                    matches.append(
                        "local_url_target"
                    )
                    continue

                if host_lower.endswith(".local"):
                    matches.append(
                        "local_url_target"
                    )
                    continue

                try:
                    address = ipaddress.ip_address(
                        host_lower
                    )

                    if (
                        address.is_private
                        or address.is_loopback
                        or address.is_link_local
                        or address.is_reserved
                        or address.is_multicast
                    ):
                        matches.append(
                            "private_url_target"
                        )

                except ValueError:
                    pass

            except ValueError:
                matches.append(
                    "malformed_url"
                )

        return matches


# ============================================================
# ISOLATED HONEYPOT
# ============================================================

class IsolatedHoneypot:
    """
    Defensive honeypot recorder.

    It records bounded metadata only.

    It NEVER executes, evaluates, connects, scans, or reads
    attacker-controlled resources.
    """

    MAX_SAMPLE_LENGTH = 512

    def __init__(
        self,
        audit_logger: SecurityAuditLogger = AUDIT_LOGGER,
    ) -> None:
        self.audit_logger = audit_logger

    def capture(
        self,
        source: str,
        service: str,
        payload: str = "",
    ) -> None:
        source_id = _source_key(source)

        sample = str(
            payload or ""
        )[: self.MAX_SAMPLE_LENGTH]

        sample_hash = hashlib.sha256(
            sample.encode(
                "utf-8",
                errors="replace",
            )
        ).hexdigest()

        self.audit_logger.record(
            event="honeypot_hit",
            source=source_id,
            severity="medium",
            service=str(service)[:128],
            payload_length=len(sample),
            payload_sha256=sample_hash,
        )


# ============================================================
# SECURITY SHIELD
# ============================================================

class SecurityShield:
    """Combined defensive security controller."""

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        block_list: TemporaryBlockList | None = None,
        threat_detector: ThreatDetector | None = None,
        audit_logger: SecurityAuditLogger | None = None,
        honeypot: IsolatedHoneypot | None = None,
    ) -> None:
        self.rate_limiter = (
            rate_limiter
            if rate_limiter is not None
            else RateLimiter()
        )

        self.block_list = (
            block_list
            if block_list is not None
            else TemporaryBlockList()
        )

        self.threat_detector = (
            threat_detector
            if threat_detector is not None
            else ThreatDetector()
        )

        self.audit_logger = (
            audit_logger
            if audit_logger is not None
            else AUDIT_LOGGER
        )

        self.honeypot = (
            honeypot
            if honeypot is not None
            else HardenedIsolatedHoneypot()
        )

    def inspect_request(
        self,
        source: str,
        text: str,
    ) -> dict[str, Any]:
        source_id = _source_key(source)

        # Defensive housekeeping.
        self.rate_limiter.cleanup()
        self.block_list.cleanup()

        if not isinstance(text, str):
            self.audit_logger.record(
                event="invalid_request",
                source=source_id,
                severity="high",
                reason="non_string_input",
            )

            return {
                "allowed": False,
                "reason": "invalid_input",
            }

        if len(text) > MAX_PROMPT_LENGTH:
            self.audit_logger.record(
                event="oversized_request",
                source=source_id,
                severity="high",
                length=len(text),
            )

            self.block_list.block(
                source
            )

            return {
                "allowed": False,
                "reason": "input_too_large",
            }

        normalized = _normalize_text(text)

        if self.block_list.is_blocked(source):
            self.audit_logger.record(
                event="blocked_request",
                source=source_id,
                severity="high",
                reason="temporary_block",
            )

            return {
                "allowed": False,
                "reason": "temporarily_blocked",
            }

        if not self.rate_limiter.allow(source):
            self.block_list.block(source)

            self.audit_logger.record(
                event="rate_limit_exceeded",
                source=source_id,
                severity="high",
                action="temporary_block",
            )

            if hasattr(
                self.honeypot,
                "capture_hashed",
            ):
                self.honeypot.capture_hashed(
                    source_hash_value=source_id,
                    service="rate-limit-honeypot",
                )
            else:
                self.honeypot.capture(
                    source=source,
                    service="rate-limit-honeypot",
                )

            return {
                "allowed": False,
                "reason": "rate_limit_exceeded",
            }

        result = self.threat_detector.inspect(
            normalized
        )

        if result["suspicious"]:
            self.audit_logger.record(
                event="suspicious_request",
                source=source_id,
                severity=result["severity"],
                matches=result["matches"],
            )

            if result["severity"] == "high":
                self.block_list.block(
                    source
                )

                if hasattr(
                    self.honeypot,
                    "capture_hashed",
                ):
                    self.honeypot.capture_hashed(
                        source_hash_value=source_id,
                        service="threat-honeypot",
                    )
                else:
                    self.honeypot.capture(
                        source=source,
                        service="threat-honeypot",
                    )

                return {
                    "allowed": False,
                    "reason": "threat_detected",
                    "severity": "high",
                    "matches": result["matches"],
                }

        return {
            "allowed": True,
            "reason": "accepted",
            "severity": result["severity"],
            "matches": result["matches"],
            "sanitized": normalized,
        }


# ============================================================
# GLOBAL SHIELD
# ============================================================

SECURITY_SHIELD = SecurityShield()

# Defensive deception stack:
# SecurityShield -> Hardened Honeypot -> SpiderWeb.
#
# No listener, socket, scanner, payload execution, or remote connection
# is created here. SpiderWeb is an in-process decoy graph only.
SECURITY_SPIDER_WEB = SpiderWeb()
attach_spider_web_to_shield(
    SECURITY_SHIELD,
    SECURITY_SPIDER_WEB,
)

# Forward metadata-only Honeypot events into the in-process decoy graph.
# The callback receives only the safe event dictionary.
SECURITY_SHIELD.honeypot.on_event = (
    SECURITY_SPIDER_WEB.observe_honeypot_event
)


# ============================================================
# PUBLIC PROMPT SANITIZATION API
# ============================================================

def validate_and_sanitize_prompt(
    prompt: str,
) -> str:
    """
    Sanitize transport-level input.

    This function does not execute or interpret the prompt.
    It removes unsafe transport characters and bounds length.
    """

    if not isinstance(prompt, str):
        raise TypeError(
            "prompt must be a string"
        )

    cleaned = _normalize_text(prompt)

    if len(cleaned) > MAX_PROMPT_LENGTH:
        cleaned = cleaned[:MAX_PROMPT_LENGTH]

    return cleaned


def validate_prompt(
    prompt: str,
) -> dict[str, Any]:
    """
    Validate a prompt against the defensive shield.

    Returns structured security information.
    """
    sanitized = validate_and_sanitize_prompt(
        prompt
    )

    result = SECURITY_SHIELD.inspect_request(
        source="prompt-validation",
        text=sanitized,
    )

    return result


def is_safe_prompt(
    prompt: str,
) -> bool:
    """Return True only when the defensive shield accepts the prompt."""
    result = validate_prompt(prompt)

    return bool(
        result.get("allowed", False)
    )


__all__ = [
    "MAX_PROMPT_LENGTH",
    "RATE_LIMIT_WINDOW",
    "RATE_LIMIT_MAX_REQUESTS",
    "DEFAULT_BLOCK_SECONDS",
    "AUDIT_LOG_PATH",
    "SecurityEvent",
    "SecurityAuditLogger",
    "RateLimiter",
    "TemporaryBlockList",
    "ThreatDetector",
    "IsolatedHoneypot",
    "SecurityShield",
    "SECURITY_SHIELD",
    "validate_and_sanitize_prompt",
    "validate_prompt",
    "is_safe_prompt",
]
