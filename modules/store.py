"""
SQLite ticket store: tickets, their message thread, an audit log of events, and agents.
Each call opens its own connection, so it is safe from Streamlit's threads.
"""
import datetime
import json
import sqlite3
import uuid
from pathlib import Path

DB_PATH = "data/support.db"

STATUSES = ["Open", "In Progress", "Waiting on Customer", "Resolved"]
OPEN_STATUSES = STATUSES[:3]
PRIORITIES = ["High", "Medium", "Low"]
# Fields agents may change; each change is written to the events table
EDITABLE_FIELDS = {"status", "assignee", "category", "priority", "needs_review", "human_reviewed", "ai_outcome"}
# AI solver result: pending = answered, waiting for the customer's verdict
AI_OUTCOMES = [None, "pending", "solved", "escalated"]

DEMO_AGENTS = [
    ("Alex Kim", "Technical"), ("Sam Lee", "Billing"), ("Linh Tran", "Logistics"),
    ("Maria Garcia", "Returns"), ("Duc Nguyen", "Account Management"),
    ("Priya Shah", "Sales"), ("Jordan Park", "General Inquiry"),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    customer_name TEXT,
    customer_email TEXT,
    tier TEXT,
    subject TEXT,
    content TEXT NOT NULL,
    category TEXT,
    confidence REAL,
    probabilities TEXT,
    sentiment TEXT,
    priority TEXT,
    status TEXT NOT NULL DEFAULT 'Open',
    assignee TEXT,
    needs_review INTEGER NOT NULL DEFAULT 0,
    human_reviewed INTEGER NOT NULL DEFAULT 0,
    routed_by_correction TEXT,
    ai_outcome TEXT,
    channel TEXT NOT NULL DEFAULT 'email',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    first_response_at TEXT,
    resolved_at TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL REFERENCES tickets(id),
    author_type TEXT NOT NULL CHECK (author_type IN ('customer', 'agent', 'ai', 'system')),
    author TEXT,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL REFERENCES tickets(id),
    actor TEXT,
    type TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agents (
    name TEXT PRIMARY KEY,
    email TEXT,
    department TEXT
);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_category ON tickets(category);
CREATE INDEX IF NOT EXISTS idx_messages_ticket ON messages(ticket_id);
CREATE INDEX IF NOT EXISTS idx_events_ticket ON events(ticket_id);
"""


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")  # readers don't block the writer
        conn.executescript(SCHEMA)
        if "ai_outcome" not in {r["name"] for r in conn.execute("PRAGMA table_info(tickets)")}:
            conn.execute("ALTER TABLE tickets ADD COLUMN ai_outcome TEXT")  # databases from before Phase 3
        conn.executemany(
            "INSERT OR IGNORE INTO agents (name, email, department) VALUES (?, ?, ?)",
            [(name, f"{name.split()[0].lower()}@example.com", dept) for name, dept in DEMO_AGENTS],
        )


def triage_fields(processed: dict) -> dict:
    """Map TicketProcessor.process_ticket() output to ticket columns."""
    validation = processed.get("routing_validation", {})
    return {
        "category": processed.get("category"),
        "confidence": processed.get("confidence_score"),
        "probabilities": json.dumps(processed.get("category_probabilities") or {}),
        "sentiment": processed.get("sentiment"),
        "priority": validation.get("priority", "Medium"),
        "needs_review": int(bool(validation.get("requires_human_review"))),
        "routed_by_correction": json.dumps(processed["routed_by_correction"]) if processed.get("routed_by_correction") else None,
    }


def _insert_ticket(conn, ticket_id, customer_name, customer_email, subject, content, triage,
                   channel, status, created_at, tier=None) -> None:
    conn.execute(
        """INSERT INTO tickets (id, customer_name, customer_email, tier, subject, content, category, confidence,
               probabilities, sentiment, priority, status, needs_review, routed_by_correction, channel,
               created_at, updated_at, resolved_at)
           VALUES (:id, :customer_name, :customer_email, :tier, :subject, :content, :category, :confidence,
               :probabilities, :sentiment, :priority, :status, :needs_review, :routed_by_correction, :channel,
               :created_at, :created_at, :resolved_at)""",
        {"id": ticket_id, "customer_name": customer_name, "customer_email": customer_email, "tier": tier,
         "subject": subject, "content": content, "status": status, "channel": channel, "created_at": created_at,
         "resolved_at": created_at if status == "Resolved" else None, **triage},
    )
    conn.execute(
        "INSERT INTO messages (ticket_id, author_type, author, body, created_at) VALUES (?, 'customer', ?, ?, ?)",
        (ticket_id, customer_name, content, created_at),
    )
    conn.execute(
        "INSERT INTO events (ticket_id, actor, type, detail, created_at) VALUES (?, 'AI', 'created', ?, ?)",
        (ticket_id, f"Routed to {triage.get('category')} ({(triage.get('confidence') or 0):.0%} confidence), "
                    f"priority {triage.get('priority')}", created_at),
    )


def create_ticket(customer_name: str, customer_email: str | None, subject: str | None, content: str,
                  triage: dict, channel: str = "web", db_path: str = DB_PATH) -> str:
    """Create a ticket with its first customer message. Returns the new ticket id."""
    if not content or not content.strip():
        raise ValueError("Ticket content is required")
    ticket_id = f"TK{uuid.uuid4().hex[:8].upper()}"
    with connect(db_path) as conn:
        _insert_ticket(conn, ticket_id, customer_name, customer_email, subject, content.strip(), triage,
                       channel, "Open", now())
    return ticket_id


def seed_from_processed(processed_tickets: list[dict], db_path: str = DB_PATH) -> int:
    """Load the processed demo tickets (from tickets.csv) if the store is empty. Returns rows added."""
    status_map = {"Open": "Open", "Pending": "Waiting on Customer", "Closed": "Resolved"}
    with connect(db_path) as conn:
        if conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]:
            return 0
        for t in processed_tickets:
            _insert_ticket(
                conn, str(t["id"]), t.get("customer") or t.get("customer_name"), None, None,
                str(t["email_content"]), triage_fields(t), "email",
                status_map.get(t.get("status"), "Open"),
                str(t.get("timestamp") or now()).replace(" ", "T"), tier=t.get("tier"),
            )
    return len(processed_tickets)


def _ticket_dict(row: sqlite3.Row) -> dict:
    t = dict(row)
    t["probabilities"] = json.loads(t["probabilities"] or "{}")
    t["routed_by_correction"] = json.loads(t["routed_by_correction"]) if t["routed_by_correction"] else None
    return t


def ticket_count(db_path: str = DB_PATH) -> int:
    with connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]


def get_ticket(ticket_id: str, db_path: str = DB_PATH) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    return _ticket_dict(row) if row else None


def list_tickets(statuses: list[str] | None = None, category: str | None = None, assignee: str | None = None,
                 needs_review: bool | None = None, search: str | None = None, db_path: str = DB_PATH) -> list[dict]:
    """Queue view, highest priority first, then oldest first. assignee='' means unassigned."""
    where, params = [], []
    if statuses:
        where.append(f"status IN ({','.join('?' * len(statuses))})")
        params += statuses
    if category:
        where.append("category = ?")
        params.append(category)
    if assignee == "":
        where.append("assignee IS NULL")
    elif assignee:
        where.append("assignee = ?")
        params.append(assignee)
    if needs_review is not None:
        where.append("needs_review = ?")
        params.append(int(needs_review))
    if search:
        where.append("(id LIKE ? OR content LIKE ? OR customer_name LIKE ?)")
        params += [f"%{search}%"] * 3
    sql = "SELECT * FROM tickets" + (" WHERE " + " AND ".join(where) if where else "")
    sql += " ORDER BY CASE priority WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END, created_at"
    with connect(db_path) as conn:
        return [_ticket_dict(r) for r in conn.execute(sql, params)]


def get_messages(ticket_id: str, db_path: str = DB_PATH) -> list[dict]:
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM messages WHERE ticket_id = ? ORDER BY created_at, id", (ticket_id,))]


def get_events(ticket_id: str, db_path: str = DB_PATH) -> list[dict]:
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM events WHERE ticket_id = ? ORDER BY created_at, id", (ticket_id,))]


def list_agents(db_path: str = DB_PATH) -> list[dict]:
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM agents ORDER BY name")]


def add_message(ticket_id: str, author_type: str, author: str, body: str, db_path: str = DB_PATH) -> None:
    """Append to the thread. The first agent/AI reply sets first_response_at."""
    if not body or not body.strip():
        raise ValueError("Message body is required")
    ts = now()
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO messages (ticket_id, author_type, author, body, created_at) VALUES (?, ?, ?, ?, ?)",
            (ticket_id, author_type, author, body.strip(), ts),
        )
        conn.execute("UPDATE tickets SET updated_at = ? WHERE id = ?", (ts, ticket_id))
        if author_type in ("agent", "ai"):
            conn.execute("UPDATE tickets SET first_response_at = ? WHERE id = ? AND first_response_at IS NULL",
                         (ts, ticket_id))
        conn.execute(
            "INSERT INTO events (ticket_id, actor, type, detail, created_at) VALUES (?, ?, 'message', ?, ?)",
            (ticket_id, author, f"{author_type} replied", ts),
        )


def update_ticket(ticket_id: str, actor: str, db_path: str = DB_PATH, **fields) -> list[str]:
    """Change editable fields, logging one event per real change. Returns the change descriptions."""
    unknown = set(fields) - EDITABLE_FIELDS
    if unknown:
        raise ValueError(f"Fields not editable: {sorted(unknown)}")
    if "status" in fields and fields["status"] not in STATUSES:
        raise ValueError(f"Unknown status: {fields['status']}")
    if "ai_outcome" in fields and fields["ai_outcome"] not in AI_OUTCOMES:
        raise ValueError(f"Unknown AI outcome: {fields['ai_outcome']}")
    ts = now()
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if row is None:
            raise KeyError(ticket_id)
        changes = []
        for field, value in fields.items():
            if row[field] == value:
                continue
            conn.execute(f"UPDATE tickets SET {field} = ?, updated_at = ? WHERE id = ?", (value, ts, ticket_id))
            changes.append(f"{field}: {row[field]} → {value}")
            conn.execute(
                "INSERT INTO events (ticket_id, actor, type, detail, created_at) VALUES (?, ?, ?, ?, ?)",
                (ticket_id, actor, field, f"{row[field]} → {value}", ts),
            )
        if fields.get("status") == "Resolved" and row["status"] != "Resolved":
            conn.execute("UPDATE tickets SET resolved_at = ? WHERE id = ?", (ts, ticket_id))
        elif "status" in fields and fields["status"] != "Resolved" and row["status"] == "Resolved":
            conn.execute("UPDATE tickets SET resolved_at = NULL WHERE id = ?", (ticket_id,))  # reopened
    return changes


def get_customer_ticket(ticket_id: str, email: str, db_path: str = DB_PATH) -> dict | None:
    """Portal lookup: the ticket only if the email matches the one it was filed with."""
    ticket = get_ticket(ticket_id.strip().upper(), db_path)
    if ticket and ticket["customer_email"] and ticket["customer_email"].lower() == email.strip().lower():
        return ticket
    return None


def customer_reply(ticket_id: str, body: str, db_path: str = DB_PATH) -> None:
    """The customer answers in the portal; a ticket waiting on them or resolved goes back to Open."""
    ticket = get_ticket(ticket_id, db_path)
    if ticket is None:
        raise KeyError(ticket_id)
    add_message(ticket_id, "customer", ticket["customer_name"], body, db_path)
    if ticket["status"] in ("Waiting on Customer", "Resolved"):
        update_ticket(ticket_id, "Customer", db_path, status="Open")
    if ticket["ai_outcome"] == "pending":  # answering instead of confirming means the AI didn't solve it
        update_ticket(ticket_id, "Customer", db_path, ai_outcome="escalated")


def ai_answer(ticket_id: str, body: str, db_path: str = DB_PATH) -> None:
    """Post the AI solver's answer and wait for the customer to confirm it solved the problem."""
    add_message(ticket_id, "ai", "Support AI", body, db_path)
    update_ticket(ticket_id, "AI", db_path, status="Waiting on Customer", ai_outcome="pending")


def customer_verdict(ticket_id: str, solved: bool, db_path: str = DB_PATH) -> None:
    """The customer's answer to "Did this solve your problem?" on an AI reply."""
    if solved:
        update_ticket(ticket_id, "Customer", db_path, status="Resolved", ai_outcome="solved")
    else:
        add_message(ticket_id, "system", "Customer", "Customer said the AI answer did not solve the problem.", db_path)
        update_ticket(ticket_id, "Customer", db_path, status="Open", ai_outcome="escalated")
