from neura.security.honeypot import IsolatedHoneypot
from neura.security.spider_web import SpiderWeb


def test_honeypot_max_events():
    print("\n=== HONEYPOT MAX_EVENTS ===")

    hp = IsolatedHoneypot(
        ttl_seconds=60,
        cleanup_interval=60,
        max_events=3,
    )

    try:
        for i in range(5):
            hp.capture(f"source-{i}", service="test")

        events = hp.events()

        print("count =", len(events))
        print("hashes =", [e["source_hash"] for e in events])

        assert len(events) == 3
        assert all("source_hash" in e for e in events)

        print("HONEYPOT MAX_EVENTS EVICTION: PASS")
    finally:
        hp.close()


def test_spider_web_max_sessions():
    print("\n=== SPIDER WEB MAX_SESSIONS ===")

    sw = SpiderWeb(
        ttl_seconds=60,
        cleanup_interval=60,
        max_sessions=3,
        max_transitions=100,
    )

    try:
        for i in range(5):
            sw.enter(f"source-{i}", "decoy-entry")

        sessions = []

        for i in range(5):
            session = sw.session(f"source-{i}")
            if session is not None:
                sessions.append(session)

        print("active_sessions =", len(sessions))
        print("remaining =", [s["source_hash"] for s in sessions])

        assert len(sessions) == 3

        print("SPIDER WEB MAX_SESSIONS EVICTION: PASS")
    finally:
        sw.close()


def test_spider_web_max_transitions():
    print("\n=== SPIDER WEB MAX_TRANSITIONS ===")

    sw = SpiderWeb(
        ttl_seconds=60,
        cleanup_interval=60,
        max_sessions=10,
        max_transitions=3,
    )

    try:
        source = "transition-test"

        sw.enter(source, "decoy-entry")
        sw.transition(source, "decoy-api")
        sw.transition(source, "decoy-config")
        sw.transition(source, "decoy-database")

        transitions = sw.transitions()

        print("transition_count =", len(transitions))

        for transition in transitions:
            print(transition)

        assert len(transitions) <= 3

        print("SPIDER WEB MAX_TRANSITIONS BOUND: PASS")
    finally:
        sw.close()


def test_expired_session_transition_cleanup():
    print("\n=== EXPIRED SESSION / TRANSITION CLEANUP ===")

    sw = SpiderWeb(
        ttl_seconds=0.01,
        cleanup_interval=60,
        max_sessions=10,
        max_transitions=10,
    )

    try:
        sw.enter("cleanup-source", "decoy-entry")
        sw.transition("cleanup-source", "decoy-api")

        import time
        time.sleep(0.05)

        removed = sw.cleanup()

        print("removed =", removed)
        print("session =", sw.session("cleanup-source"))
        print("transitions =", sw.transitions())

        assert sw.session("cleanup-source") is None
        assert sw.transitions() == []

        print("EXPIRED SESSION/TRANSITION CLEANUP: PASS")
    finally:
        sw.close()


if __name__ == "__main__":
    print("\n=== SECURITY BOUNDS TEST ===")

    test_honeypot_max_events()
    test_spider_web_max_sessions()
    test_spider_web_max_transitions()
    test_expired_session_transition_cleanup()

    print("ALL BOUNDS TESTS: PASS")
