"""
Neura Defensive Spider Web
==========================

An isolated defensive deception graph.

The Spider Web represents decoy nodes and transitions between them.
It does NOT:
- open network ports
- accept external connections
- execute payloads
- evaluate code
- scan remote systems
- contact attackers
- store raw payloads
- store secrets

Only source hashes and bounded event metadata are retained.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Any


DEFAULT_SESSION_TTL = 900.0
DEFAULT_CLEANUP_INTERVAL = 60.0
DEFAULT_MAX_SESSIONS = 2000
DEFAULT_MAX_TRANSITIONS = 10000

RISK_LOW = 0
RISK_MEDIUM = 1
RISK_HIGH = 2
RISK_CRITICAL = 3


def source_hash(source: str) -> str:
    """Return a stable non-reversible source identifier."""
    value = str(source or "unknown").strip()

    return hashlib.sha256(
        value.encode(
            "utf-8",
            errors="replace",
        )
    ).hexdigest()[:32]


@dataclass(frozen=True)
class DecoyNode:
    node_id: str
    node_type: str
    next_nodes: tuple[str, ...]


@dataclass(frozen=True)
class SpiderTransition:
    source_hash: str
    from_node: str
    to_node: str
    timestamp: float
    count: int
    risk_level: int
    expires_at: float


@dataclass
class SpiderSession:
    source_hash: str
    current_node: str
    first_seen: float
    last_seen: float
    expires_at: float
    transitions: int
    risk_level: int


class SpiderWeb:
    """
    Isolated graph of defensive decoy nodes.

    The graph exists only in memory and has no network listener.
    """

    DEFAULT_NODES = (
        DecoyNode(
            "decoy-entry",
            "entry",
            (
                "decoy-login",
                "decoy-api",
            ),
        ),
        DecoyNode(
            "decoy-login",
            "authentication-decoy",
            (
                "decoy-config",
                "decoy-session",
            ),
        ),
        DecoyNode(
            "decoy-api",
            "api-decoy",
            (
                "decoy-config",
                "decoy-database",
            ),
        ),
        DecoyNode(
            "decoy-config",
            "configuration-decoy",
            (
                "decoy-database",
                "decoy-admin",
            ),
        ),
        DecoyNode(
            "decoy-session",
            "session-decoy",
            (
                "decoy-admin",
            ),
        ),
        DecoyNode(
            "decoy-database",
            "database-decoy",
            (
                "decoy-admin",
            ),
        ),
        DecoyNode(
            "decoy-admin",
            "administration-decoy",
            (),
        ),
    )

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_SESSION_TTL,
        cleanup_interval: float = DEFAULT_CLEANUP_INTERVAL,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_transitions: int = DEFAULT_MAX_TRANSITIONS,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(
                "ttl_seconds must be positive"
            )

        if cleanup_interval <= 0:
            raise ValueError(
                "cleanup_interval must be positive"
            )

        if max_sessions <= 0:
            raise ValueError(
                "max_sessions must be positive"
            )

        if max_transitions <= 0:
            raise ValueError(
                "max_transitions must be positive"
            )

        self.ttl_seconds = float(ttl_seconds)
        self.cleanup_interval = float(
            cleanup_interval
        )
        self.max_sessions = int(
            max_sessions
        )
        self.max_transitions = int(
            max_transitions
        )

        self._nodes = {
            node.node_id: node
            for node in self.DEFAULT_NODES
        }

        self._sessions: dict[
            str,
            SpiderSession,
        ] = {}

        self._transitions: list[
            SpiderTransition
        ] = []

        self._transition_counts: dict[
            tuple[str, str, str],
            int,
        ] = {}

        self._lock = threading.RLock()
        self._stop_event = threading.Event()

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="neura-spider-web-cleaner",
            daemon=True,
        )

        self._cleanup_thread.start()

    # ========================================================
    # NODE GRAPH
    # ========================================================

    def nodes(self) -> list[dict[str, Any]]:
        """Return public metadata for decoy nodes."""
        with self._lock:
            return [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "next_nodes": list(
                        node.next_nodes
                    ),
                }
                for node in self._nodes.values()
            ]

    def has_node(
        self,
        node_id: str,
    ) -> bool:
        return str(node_id) in self._nodes

    # ========================================================
    # SESSION
    # ========================================================

    def _get_or_create_session(
        self,
        source: str,
        now: float,
    ) -> SpiderSession:
        return self._get_or_create_session_hashed(
            source_hash(source),
            now,
        )

    def _get_or_create_session_hashed(
        self,
        key: str,
        now: float,
    ) -> SpiderSession:
        session = self._sessions.get(key)

        if session is not None:
            session.last_seen = now
            session.expires_at = (
                now + self.ttl_seconds
            )
            return session

        if len(self._sessions) >= self.max_sessions:
            self._evict_oldest_locked()

        session = SpiderSession(
            source_hash=key,
            current_node="decoy-entry",
            first_seen=now,
            last_seen=now,
            expires_at=now + self.ttl_seconds,
            transitions=0,
            risk_level=RISK_LOW,
        )

        self._sessions[key] = session

        return session

    def _evict_oldest_locked(self) -> None:
        if not self._sessions:
            return

        oldest_key = min(
            self._sessions,
            key=lambda key: (
                self._sessions[key].last_seen
            ),
        )

        self._sessions.pop(
            oldest_key,
            None,
        )

    # ========================================================
    # TRANSITIONS
    # ========================================================

    def enter(
        self,
        source: str,
        node_id: str = "decoy-entry",
    ) -> dict[str, Any]:
        """
        Record entry into a decoy node.

        This only changes in-memory deception state.
        """
        now = time.time()

        with self._lock:
            self._cleanup_locked(now)

            if node_id not in self._nodes:
                return {
                    "accepted": False,
                    "reason": "unknown_decoy_node",
                }

            session = self._get_or_create_session(
                source,
                now,
            )

            if (
                session.current_node
                != node_id
            ):
                return self._transition_locked(
                    session,
                    node_id,
                    now,
                )

            return self._snapshot_locked(
                session,
                transition=False,
            )

    def transition(
        self,
        source: str,
        to_node: str,
    ) -> dict[str, Any]:
        """
        Record a logical transition between decoy nodes.

        The destination must be explicitly present in the graph.
        """
        now = time.time()

        with self._lock:
            self._cleanup_locked(now)

            if to_node not in self._nodes:
                return {
                    "accepted": False,
                    "reason": "unknown_decoy_node",
                }

            session = self._get_or_create_session(
                source,
                now,
            )

            return self._transition_locked(
                session,
                to_node,
                now,
            )

    def _transition_locked(
        self,
        session: SpiderSession,
        to_node: str,
        now: float,
    ) -> dict[str, Any]:
        from_node = session.current_node

        allowed_targets = self._nodes[
            from_node
        ].next_nodes

        if to_node not in allowed_targets:
            # Do not create arbitrary graph edges.
            session.risk_level = min(
                RISK_CRITICAL,
                session.risk_level + 1,
            )

            return self._snapshot_locked(
                session,
                transition=False,
                accepted=False,
                reason="invalid_transition",
            )

        transition_key = (
            session.source_hash,
            from_node,
            to_node,
        )

        count = (
            self._transition_counts.get(
                transition_key,
                0,
            )
            + 1
        )

        self._transition_counts[
            transition_key
        ] = count

        session.current_node = to_node
        session.last_seen = now
        session.expires_at = (
            now + self.ttl_seconds
        )
        session.transitions += 1

        # Repeated movement through the same edge
        # progressively raises risk.
        if count >= 5:
            session.risk_level = max(
                session.risk_level,
                RISK_CRITICAL,
            )
        elif count >= 3:
            session.risk_level = max(
                session.risk_level,
                RISK_HIGH,
            )
        elif count >= 2:
            session.risk_level = max(
                session.risk_level,
                RISK_MEDIUM,
            )

        if len(self._transitions) >= (
            self.max_transitions
        ):
            del self._transitions[
                : max(
                    1,
                    len(self._transitions)
                    - self.max_transitions
                    + 1,
                )
            ]

        self._transitions.append(
            SpiderTransition(
                source_hash=session.source_hash,
                from_node=from_node,
                to_node=to_node,
                timestamp=now,
                count=count,
                risk_level=session.risk_level,
                expires_at=session.expires_at,
            )
        )

        return self._snapshot_locked(
            session,
            transition=True,
            accepted=True,
        )

    # ========================================================
    # HONEYPOT EVENT INTEGRATION
    # ========================================================

    def observe_honeypot_event(
        self,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert a Honeypot metadata event into a graph observation.

        The Honeypot already provides a source_hash, so it is used
        directly rather than hashed a second time.

        Payload/content/secret fields are ignored.
        """
        source_id = str(
            event.get(
                "source_hash",
                "unknown",
            )
        )[:32]

        service = str(
            event.get(
                "service",
                "honeypot",
            )
        )

        mapping = {
            "rate-limit-honeypot":
                "decoy-entry",
            "threat-honeypot":
                "decoy-api",
            "login-honeypot":
                "decoy-login",
            "config-honeypot":
                "decoy-config",
            "database-honeypot":
                "decoy-database",
        }

        node = mapping.get(
            service,
            "decoy-entry",
        )

        return self.enter_hashed(
            source_id,
            node,
        )

    def enter_hashed(
        self,
        source_id: str,
        node_id: str = "decoy-entry",
    ) -> dict[str, Any]:
        """
        Enter the graph using an already-generated source hash.

        This method is intentionally separate from enter() so that
        Honeypot source hashes are never hashed twice.
        """
        now = time.time()

        source_id = str(
            source_id or "unknown"
        )[:32]

        with self._lock:
            self._cleanup_locked(now)

            if node_id not in self._nodes:
                return {
                    "accepted": False,
                    "reason": "unknown_decoy_node",
                }

            session = self._get_or_create_session_hashed(
                source_id,
                now,
            )

            if session.current_node != node_id:
                return self._transition_locked(
                    session,
                    node_id,
                    now,
                )

            return self._snapshot_locked(
                session,
                transition=False,
            )

    # ========================================================
    # SNAPSHOT
    # ========================================================

    @staticmethod
    def _risk_name(
        risk_level: int,
    ) -> str:
        names = {
            RISK_LOW: "low",
            RISK_MEDIUM: "medium",
            RISK_HIGH: "high",
            RISK_CRITICAL: "critical",
        }

        return names.get(
            risk_level,
            "unknown",
        )

    def _snapshot_locked(
        self,
        session: SpiderSession,
        transition: bool,
        accepted: bool = True,
        reason: str = "accepted",
    ) -> dict[str, Any]:
        return {
            "accepted": accepted,
            "reason": reason,
            "source_hash": session.source_hash,
            "current_node": session.current_node,
            "transitions": session.transitions,
            "risk_level": self._risk_name(
                session.risk_level
            ),
            "transition_recorded": transition,
            "expires_at": session.expires_at,
        }

    # ========================================================
    # REPORTING
    # ========================================================

    def transitions(
        self,
    ) -> list[dict[str, Any]]:
        now = time.time()

        with self._lock:
            self._cleanup_locked(now)

            return [
                {
                    "source_hash": item.source_hash,
                    "from_node": item.from_node,
                    "to_node": item.to_node,
                    "timestamp": item.timestamp,
                    "count": item.count,
                    "risk_level": self._risk_name(
                        item.risk_level
                    ),
                    "expires_at": item.expires_at,
                }
                for item in self._transitions
            ]

    def session(
        self,
        source: str,
    ) -> dict[str, Any] | None:
        key = source_hash(source)
        now = time.time()

        with self._lock:
            self._cleanup_locked(now)

            item = self._sessions.get(key)

            if item is None:
                return None

            return {
                "source_hash": item.source_hash,
                "current_node": item.current_node,
                "first_seen": item.first_seen,
                "last_seen": item.last_seen,
                "expires_at": item.expires_at,
                "transitions": item.transitions,
                "risk_level": self._risk_name(
                    item.risk_level
                ),
            }

    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(self) -> int:
        now = time.time()

        with self._lock:
            before = (
                len(self._sessions)
                + len(self._transitions)
            )

            self._cleanup_locked(now)

            after = (
                len(self._sessions)
                + len(self._transitions)
            )

            return before - after

    def _cleanup_locked(
        self,
        now: float,
    ) -> None:
        expired_sources = {
            key
            for key, session in self._sessions.items()
            if session.expires_at <= now
        }

        for key in expired_sources:
            self._sessions.pop(
                key,
                None,
            )

        self._transitions = [
            item
            for item in self._transitions
            if item.expires_at > now
        ]

        if expired_sources:
            stale_edges = [
                key
                for key in self._transition_counts
                if key[0] in expired_sources
            ]

            for key in stale_edges:
                self._transition_counts.pop(
                    key,
                    None,
                )

    def _cleanup_loop(self) -> None:
        while not self._stop_event.wait(
            self.cleanup_interval
        ):
            try:
                self.cleanup()
            except Exception:
                pass

    # ========================================================
    # LIFECYCLE
    # ========================================================

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
# SHIELD INTEGRATION
# ============================================================

def attach_to_shield(
    shield: Any,
    spider_web: SpiderWeb | None = None,
) -> SpiderWeb:
    """
    Attach Spider Web to an existing SecurityShield.

    The Shield remains the enforcement layer.
    Spider Web is telemetry/deception only.
    """
    if shield is None:
        raise ValueError(
            "shield must not be None"
        )

    if spider_web is None:
        spider_web = SpiderWeb()

    shield.spider_web = spider_web

    return spider_web


__all__ = [
    "DEFAULT_SESSION_TTL",
    "DEFAULT_CLEANUP_INTERVAL",
    "DEFAULT_MAX_SESSIONS",
    "DEFAULT_MAX_TRANSITIONS",
    "RISK_LOW",
    "RISK_MEDIUM",
    "RISK_HIGH",
    "RISK_CRITICAL",
    "source_hash",
    "DecoyNode",
    "SpiderTransition",
    "SpiderSession",
    "SpiderWeb",
    "attach_to_shield",
]
