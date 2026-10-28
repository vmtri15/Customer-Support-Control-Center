"""Self-check for the SQLite ticket store. Uses a temporary database. Run: python test_store.py"""
import os
import sqlite3
import tempfile

from modules import store

db = os.path.join(tempfile.mkdtemp(), "support.db")
store.init_db(db)
store.init_db(db)  # idempotent
assert len(store.list_agents(db)) == len(store.DEMO_AGENTS)

triage = {"category": "Billing", "confidence": 0.9, "probabilities": "{}", "sentiment": "Neutral",
          "priority": "Medium", "needs_review": 0, "routed_by_correction": None}

# Seeding: maps CSV statuses, only runs on an empty store
processed = [
    {"id": "T1", "customer": "Ann", "email_content": "Refund please", "status": "Open", "timestamp": "2026-10-01 10:00:00",
     "category": "Billing", "confidence_score": 0.9, "category_probabilities": {"Billing": 0.9}, "sentiment": "Neutral",
     "routing_validation": {"priority": "High", "requires_human_review": False}},
    {"id": "T2", "customer": "Bo", "email_content": "Where is my parcel", "status": "Closed", "timestamp": "2026-10-02 10:00:00",
     "category": "Logistics", "confidence_score": 0.4, "category_probabilities": {}, "sentiment": "Neutral",
     "routing_validation": {"priority": "Low", "requires_human_review": True}},
]
assert store.seed_from_processed(processed, db) == 2
assert store.seed_from_processed(processed, db) == 0
t2 = store.get_ticket("T2", db)
assert t2["status"] == "Resolved" and t2["resolved_at"] and t2["needs_review"] == 1
assert store.get_messages("T1", db)[0]["author_type"] == "customer"

# Queue: priority first, filters
tid = store.create_ticket("Cy", "cy@example.com", "App down", "  The app crashes  ", {**triage, "priority": "Low"}, db_path=db)
assert tid.startswith("TK") and store.get_ticket(tid, db)["content"] == "The app crashes"
queue = store.list_tickets(statuses=store.OPEN_STATUSES, db_path=db)
assert [t["id"] for t in queue] == ["T1", tid], queue
assert [t["id"] for t in store.list_tickets(search="parcel", db_path=db)] == ["T2"]
assert [t["id"] for t in store.list_tickets(assignee="", category="Logistics", db_path=db)] == ["T2"]
assert [t["id"] for t in store.list_tickets(needs_review=True, db_path=db)] == ["T2"]

# Replies set first_response_at once
store.add_message("T1", "agent", "Sam Lee", "We are on it", db)
first = store.get_ticket("T1", db)["first_response_at"]
store.add_message("T1", "agent", "Sam Lee", "Second reply", db)
assert first and store.get_ticket("T1", db)["first_response_at"] == first
assert len(store.get_messages("T1", db)) == 3

# Updates: only real changes are logged; resolve/reopen sets/clears resolved_at
assert store.update_ticket("T1", "Sam Lee", db_path=db, status="Open") == []
changes = store.update_ticket("T1", "Sam Lee", db_path=db, status="Resolved", assignee="Sam Lee")
assert len(changes) == 2 and store.get_ticket("T1", db)["resolved_at"]
store.update_ticket("T1", "Sam Lee", db_path=db, status="In Progress")
assert store.get_ticket("T1", db)["resolved_at"] is None
assert [e["type"] for e in store.get_events("T1", db)] == ["created", "message", "message", "status", "assignee", "status"]

# Validation at the boundary
for bad in [lambda: store.update_ticket("T1", "x", db_path=db, content="hacked"),
            lambda: store.update_ticket("T1", "x", db_path=db, status="Deleted"),
            lambda: store.create_ticket("A", None, None, "   ", triage, db_path=db),
            lambda: store.add_message("T1", "agent", "x", " ", db)]:
    try:
        bad()
        raise AssertionError("should have raised")
    except ValueError:
        pass
try:
    store.add_message("T1", "hacker", "x", "hi", db)
    raise AssertionError("author_type must be checked")
except sqlite3.IntegrityError:
    pass

# Portal: lookup needs the matching email; a customer reply reopens the ticket
assert store.get_customer_ticket(tid.lower(), " CY@example.com ", db)["id"] == tid
assert store.get_customer_ticket(tid, "other@example.com", db) is None
assert store.get_customer_ticket("T1", "", db) is None  # seeded tickets have no email
store.update_ticket(tid, "Sam Lee", db_path=db, status="Waiting on Customer")
store.customer_reply(tid, "Still crashing", db)
assert store.get_ticket(tid, db)["status"] == "Open"
assert store.get_messages(tid, db)[-1]["author_type"] == "customer"
assert store.get_events(tid, db)[-1]["actor"] == "Customer"

# AI solver: answer -> customer verdict; replying instead of confirming counts as escalated
store.ai_answer(tid, "Try reinstalling the app.", db)
t = store.get_ticket(tid, db)
assert t["status"] == "Waiting on Customer" and t["ai_outcome"] == "pending"
store.customer_verdict(tid, solved=True, db_path=db)
t = store.get_ticket(tid, db)
assert t["status"] == "Resolved" and t["ai_outcome"] == "solved" and t["resolved_at"]
store.ai_answer("T1", "Refund issued.", db)
store.customer_verdict("T1", solved=False, db_path=db)
assert store.get_ticket("T1", db)["status"] == "Open" and store.get_ticket("T1", db)["ai_outcome"] == "escalated"
assert store.get_messages("T1", db)[-1]["author_type"] == "system"
store.ai_answer("T2", "Tracking link sent.", db)
store.customer_reply("T2", "That link is broken", db)
assert store.get_ticket("T2", db)["ai_outcome"] == "escalated"
try:
    store.update_ticket("T1", "x", db_path=db, ai_outcome="maybe")
    raise AssertionError("ai_outcome must be validated")
except ValueError:
    pass

# Databases created before Phase 3 get the ai_outcome column
old = os.path.join(tempfile.mkdtemp(), "old.db")
with sqlite3.connect(old) as conn:
    conn.executescript(store.SCHEMA.replace("    ai_outcome TEXT,\n", ""))
store.init_db(old)
with sqlite3.connect(old) as conn:
    assert "ai_outcome" in {r[1] for r in conn.execute("PRAGMA table_info(tickets)")}

print("✅ store checks passed")
