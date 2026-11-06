"""
Neura Isolated Defensive Honeypot
=================================

A metadata-only defensive honeypot.

Security properties:
- stores source hashes, never raw source identifiers
- stores event metadata only
- never stores payloads
- never stores secrets
- never executes received content
- never evaluates code
- never opens network connections
- automatic TTL expiration
- automatic background cleanup
- bounded in-memory state
- thread-safe
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_TTL_SECONDS = 900.0
DEFAULT_CLEANUP_INTERVAL = 60.0
DEFAULT_MAX_EVENTS = 5000
MAX_SERVICE_LENGTH = 128


def source_hash(source: str) -> str:
    """
    Produce a stable non-reversible identifier.

    Raw source values are never retained by this module.
    """
    value = str(source or "unknown").strip()

    return hashlib.sha256(
        value.encode(
            "utf-8",
            errors="replace",
        )
    ).hexdigest()[:32]


@dataclass(frozen=True)
class HoneypotEvent:
    source_hash: str
    service: str
    event: str
    timestamp: float
    expires_at: float


class IsolatedHoneypot:
    """
    Metadata-only isolated honeypot.

    The honeypot is intentionally passive. It records that a source
    touched a decoy and nothing more.
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        cleanup_interval: float = DEFAULT_CLEANUP_INTERVAL,
        max_events: int = DEFAULT_MAX_EVENTS,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(
                "ttl_seconds must be positive"
            )

        if cleanup_interval <= 0:
            raise ValueError(
                "cleanup_interval must be positive"
            )

        if max_events <= 0:
            raise ValueError(
                "max_events must be positive"
            )

        self.ttl_seconds = float(ttl_seconds)
        self.cleanup_interval = float(cleanup_interval)
        self.max_events = int(max_events)
        self.on_event = on_event

        self._events: list[HoneypotEvent] = []
        self._lock = threading.RLock()
        self._stop_event = threading.Event()

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="neura-honeypot-cleaner",
            daemon=True,
        )

        self._cleanup_thread.start()

    # --------------------------------------------------------
    # Capture
    # --------------------------------------------------------

    def capture(
        self,
        source: str,
        service: str,
        payload: str = "",
        event: str = "honeypot_hit",
        **metadata: Any,
    ) -> None:
        """
        Record metadata only.

        `payload` is intentionally ignored and is NEVER stored,
        hashed, executed, parsed, or forwarded.
        """
        del payload

        now = time.time()

        record = HoneypotEvent(
            source_hash=source_hash(source),
            service=str(service)[:MAX_SERVICE_LENGTH],
            event=str(event)[:MAX_SERVICE_LENGTH],
            timestamp=now,
            expires_at=now + self.ttl_seconds,
        )

        with self._lock:
            self._cleanup_locked(now)

            self._events.append(record)

            if len(self._events) > self.max_events:
                overflow = (
                    len(self._events)
                    - self.max_events
                )

                del self._events[:overflow]

        if self.on_event is not None:
            safe_metadata = {
                "source_hash": record.source_hash,
                "service": record.service,
                "event": record.event,
                "timestamp": record.timestamp,
                "expires_at": record.expires_at,
            }

            # Only bounded scalar metadata is exposed.
            for key, value in metadata.items():
                if key in {
                    "payload",
                    "prompt",
                    "secret",
                    "password",
                    "token",
                    "cookie",
                    "authorization",
                    "content",
                }:
                    continue

                if isinstance(
                    value,
                    (str, int, float, bool),
                ):
                    safe_metadata[
                        str(key)[:64]
                    ] = value

            try:
                self.on_event(
                    safe_metadata
                )
            except Exception:
                # Security telemetry must never break the request path.
                pass

    def capture_hashed(
        self,
        source_hash_value: str,
        service: str,
        event: str = "honeypot_hit",
        **metadata: Any,
    ) -> None:
        """
        Record an already-generated source hash.

        This is used when SecurityShield has already converted
        the source identifier into a non-reversible hash.

        The hash is never hashed again.
        """
        safe_hash = str(
            source_hash_value or "unknown"
        )[:32]

        now = time.time()

        record = HoneypotEvent(
            source_hash=safe_hash,
            service=str(service)[:MAX_SERVICE_LENGTH],
            event=str(event)[:MAX_SERVICE_LENGTH],
            timestamp=now,
            expires_at=now + self.ttl_seconds,
        )

        with self._lock:
            self._cleanup_locked(now)

            self._events.append(record)

            if len(self._events) > self.max_events:
                overflow = (
                    len(self._events)
                    - self.max_events
                )

                del self._events[:overflow]

        if self.on_event is not None:
            safe_metadata = {
                "source_hash": record.source_hash,
                "service": record.service,
                "event": record.event,
                "timestamp": record.timestamp,
                "expires_at": record.expires_at,
            }

            for key, value in metadata.items():
                if key in {
                    "payload",
                    "prompt",
                    "secret",
                    "password",
                    "token",
                    "cookie",
                    "authorization",
                    "content",
                }:
                    continue

                if isinstance(
                    value,
                    (str, int, float, bool),
                ):
                    safe_metadata[
                        str(key)[:64]
                    ] = value

            try:
                self.on_event(
                    safe_metadata
                )
            except Exception:
                pass

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    def events(
        self,
    ) -> list[dict[str, Any]]:
        """Return non-sensitive active event metadata."""
        now = time.time()

        with self._lock:
            self._cleanup_locked(now)

            return [
                {
                    "source_hash": event.source_hash,
                    "service": event.service,
                    "event": event.event,
                    "timestamp": event.timestamp,
                    "expires_at": event.expires_at,
                }
                for event in self._events
            ]

    def count(
        self,
    ) -> int:
        now = time.time()

        with self._lock:
            self._cleanup_locked(now)
            return len(self._events)

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    def cleanup(self) -> int:
        """
        Remove expired events.

        Returns the number of removed records.
        """
        now = time.time()

        with self._lock:
            before = len(self._events)

            self._cleanup_locked(now)

            return before - len(self._events)

    def _cleanup_locked(
        self,
        now: float,
    ) -> None:
        self._events = [
            event
            for event in self._events
            if event.expires_at > now
        ]

    def _cleanup_loop(self) -> None:
        while not self._stop_event.wait(
            self.cleanup_interval
        ):
            try:
                self.cleanup()
            except Exception:
                # Cleaner failure must never affect the main application.
                pass

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def close(self) -> None:
        self._stop_event.set()

        thread = self._cleanup_thread

        if (
            thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(
                timeout=min(
                    self.cleanup_interval,
                    2.0,
                )
            )

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()


# ============================================================
# SECURITYSHIELD INTEGRATION
# ============================================================

def attach_to_shield(
    shield: Any,
    honeypot: IsolatedHoneypot | None = None,
) -> IsolatedHoneypot:
    """
    Attach the honeypot to an existing SecurityShield instance.

    The shield already calls `.capture()` for:
    - rate-limit violations
    - high-severity threat detections

    No offensive behavior is added.
    """
    if shield is None:
        raise ValueError(
            "shield must not be None"
        )

    if honeypot is None:
        honeypot = IsolatedHoneypot()

    shield.honeypot = honeypot

    return honeypot


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "DEFAULT_CLEANUP_INTERVAL",
    "DEFAULT_MAX_EVENTS",
    "source_hash",
    "HoneypotEvent",
    "IsolatedHoneypot",
    "attach_to_shield",
]
