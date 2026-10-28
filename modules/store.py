"""
SQLite ticket store: tickets, their message thread, an audit log of events, and agents.
Each call opens its own connection, so it is safe from Streamlit's threads.
"""
import datetime
import json
import random
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
# Response-time targets in hours, by priority: (first reply, resolution)
SLA_HOURS = {"High": (1, 8), "Medium": (4, 24), "Low": (24, 72)}

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
    csat INTEGER CHECK (csat BETWEEN 1 AND 5),
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
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(tickets)")}
        for column, ddl in [("ai_outcome", "TEXT"), ("csat", "INTEGER CHECK (csat BETWEEN 1 AND 5)")]:
            if column not in columns:  # databases created by an older version
                conn.execute(f"ALTER TABLE tickets ADD COLUMN {column} {ddl}")
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


def seed_from_processed(processed_tickets: list[dict], db_path: str = DB_PATH, days: int = 30,
                        seed: int = 42) -> int:
    """Load the processed demo tickets (from tickets.csv) if the store is empty, with a made-up but realistic
    history over the last `days`: older tickets were answered and resolved by the department's agent (mostly
    within target, some late, some rated), recent ones are still in the queue. Returns rows added."""
    rng = random.Random(seed)
    end = datetime.datetime.now().replace(microsecond=0)
    agent_for = {dept: name for name, dept in DEMO_AGENTS}
    iso = lambda dt: dt.isoformat(timespec="seconds")  # noqa: E731
    with connect(db_path) as conn:
        if conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]:
            return 0
        for t in processed_tickets:
            triage = triage_fields(t)
            first_h, resolve_h = SLA_HOURS.get(triage["priority"], SLA_HOURS["Medium"])
            late = rng.random() < 0.15
            created = end - datetime.timedelta(hours=rng.uniform(0, days * 24))
            replied = created + datetime.timedelta(hours=rng.uniform(0.05, 1) * first_h * (2.5 if late else 1))
            resolved = replied + datetime.timedelta(hours=rng.uniform(0.1, 1) * (resolve_h - first_h) * (2 if late else 1))
            if resolved <= end and rng.random() < 0.99:  # a few old tickets got stuck: overdue demo
                status = "Resolved"
            elif replied <= end:
                status = rng.choice(["In Progress", "Waiting on Customer"])
            else:
                status, replied = "Open", None
            ticket_id = str(t["id"])
            name = t.get("customer") or t.get("customer_name")
            _insert_ticket(conn, ticket_id, name, None, None, str(t["email_content"]), triage, "email",
                           status, iso(created), tier=t.get("tier"))
            if replied is None:
                continue
            agent = agent_for.get(triage["category"], DEMO_AGENTS[-1][0])
            response = t.get("response")
            body = response if isinstance(response, str) and response.strip() else \
                f"Hi {name or 'there'}, thanks for contacting us. Our {triage['category']} team is looking into this."
            csat = None
            if status == "Resolved" and rng.random() < 0.4:
                csat = rng.choice([3, 2, 1, 3, 4] if late else [5, 5, 4, 4, 3])
            conn.execute("UPDATE tickets SET assignee = ?, first_response_at = ?, resolved_at = ?, updated_at = ?, "
                         "csat = ? WHERE id = ?",
                         (agent, iso(replied), iso(resolved) if status == "Resolved" else None,
                          iso(resolved if status == "Resolved" else replied), csat, ticket_id))
            conn.executemany(
                "INSERT INTO events (ticket_id, actor, type, detail, created_at) VALUES (?, ?, ?, ?, ?)",
                [(ticket_id, agent, "assignee", f"None → {agent}", iso(replied)),
                 (ticket_id, agent, "message", "agent replied", iso(replied))]
                + ([(ticket_id, agent, "status", "Open → Resolved", iso(resolved))] if status == "Resolved" else []))
            conn.execute("INSERT INTO messages (ticket_id, author_type, author, body, created_at) "
                         "VALUES (?, 'agent', ?, ?, ?)", (ticket_id, agent, body.strip(), iso(replied)))
    return len(processed_tickets)


def sla(ticket: dict, at: datetime.datetime | None = None) -> dict:
    """Response-time targets for a ticket: due times, whether each was missed, and what is due next."""
    # ponytail: nothing is due from agents while Waiting on Customer, but that time still counts toward
    # the resolution target once the customer answers; subtract waiting time if that proves unfair
    at = at or datetime.datetime.now()
    created = datetime.datetime.fromisoformat(ticket["created_at"])
    first_h, resolve_h = SLA_HOURS.get(ticket["priority"], SLA_HOURS["Medium"])
    out = {}
    for name, hours, done in [("first_response", first_h, ticket["first_response_at"]),
                              ("resolution", resolve_h, ticket["resolved_at"])]:
        due = created + datetime.timedelta(hours=hours)
        finished = datetime.datetime.fromisoformat(done) if done else None
        out[f"{name}_due"] = due
        out[f"{name}_breached"] = (finished or at) > due
    if ticket["resolved_at"] or ticket.get("status") == "Waiting on Customer":
        out["next_due"], out["next_target"] = None, None
    elif not ticket["first_response_at"]:
        out["next_due"], out["next_target"] = out["first_response_due"], "first reply"
    else:
        out["next_due"], out["next_target"] = out["resolution_due"], "resolution"
    out["overdue"] = out["next_due"] is not None and at > out["next_due"]
    return out


def rate_ticket(ticket_id: str, rating: int, db_path: str = DB_PATH) -> None:
    """Customer satisfaction, 1-5 stars, on a resolved ticket."""
    if rating not in range(1, 6):
        raise ValueError("Rating must be 1-5")
    with connect(db_path) as conn:
        conn.execute("UPDATE tickets SET csat = ? WHERE id = ?", (rating, ticket_id))
        conn.execute("INSERT INTO events (ticket_id, actor, type, detail, created_at) "
                     "VALUES (?, 'Customer', 'csat', ?, ?)", (ticket_id, f"Rated {rating}/5", now()))


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


def _since(hours: float) -> str:
    return (datetime.datetime.now() - datetime.timedelta(hours=hours)).isoformat(timespec="seconds")


def log_llm_call(ticket_id: str, actor: str, purpose: str, db_path: str = DB_PATH) -> None:
    """Record an LLM call on the ticket's activity log; llm_calls_since() counts them for the daily cap."""
    with connect(db_path) as conn:
        conn.execute("INSERT INTO events (ticket_id, actor, type, detail, created_at) VALUES (?, ?, 'llm', ?, ?)",
                     (ticket_id, actor, purpose, now()))


def llm_calls_since(hours: float, db_path: str = DB_PATH) -> int:
    with connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM events WHERE type = 'llm' AND created_at >= ?",
                            (_since(hours),)).fetchone()[0]


def web_tickets_since(hours: float, email: str | None = None, db_path: str = DB_PATH) -> int:
    """Help Center submissions in the last `hours`, optionally from one email (case-insensitive)."""
    sql, params = "SELECT COUNT(*) FROM tickets WHERE channel = 'web' AND created_at >= ?", [_since(hours)]
    if email:
        sql += " AND lower(customer_email) = ?"
        params.append(email.strip().lower())
    with connect(db_path) as conn:
        return conn.execute(sql, params).fetchone()[0]
