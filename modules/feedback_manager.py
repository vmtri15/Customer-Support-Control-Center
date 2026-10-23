import datetime
import json
from pathlib import Path

class FeedbackManager:
    """Manages Human-in-the-loop workflows: Stores human feedback and calculates KPIs."""

    def __init__(self, corrections_path: str | Path = "data/corrections.json"):
        self.corrections_path = Path(corrections_path)
        # Automatically create the data directory if it doesn't exist
        self.corrections_path.parent.mkdir(parents=True, exist_ok=True)
        self.corrections = self._load_corrections()

    def _load_corrections(self) -> list[dict]:
        if self.corrections_path.exists():
            with self.corrections_path.open(encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_correction(self, ticket_id: str, original_dept: str, corrected_dept: str, reason: str, email_content: str = "") -> None:
        """Logs the manual correction and updates the Vector DB to facilitate system learning."""
        correction = {
            "ticket_id": ticket_id,
            "original_recommended_department": original_dept,
            "corrected_department": corrected_dept,
            "reason": reason,
            "email_content": email_content,
            "status": "Human_Reviewed"
        }
        self.corrections.append(correction)
        
        with self.corrections_path.open("w", encoding="utf-8") as f:
            json.dump(self.corrections, f, indent=4, ensure_ascii=False)
            
        # Update Vector DB: similar low-confidence tickets now follow this correction
        if email_content:
            try:
                from modules.vector_store import add_correction
                add_correction(ticket_id, email_content, corrected_dept)
                print(f"🧠 Vector DB updated: AI learned that Ticket {ticket_id} belongs to {corrected_dept}")
            except Exception as e:
                print(f"⚠️ Failed to update Vector DB: {e}")

        print(f"✅ Feedback logged for Ticket {ticket_id}: Successfully re-routed to {corrected_dept}")

    def log_reply(self, ticket_id: str, reply: str, replies_path: str | Path = "data/sent_replies.json") -> None:
        """Append the agent-approved reply (as edited) to the sent-replies log."""
        replies_path = Path(replies_path)
        replies = json.loads(replies_path.read_text(encoding="utf-8")) if replies_path.exists() else []
        replies.append({
            "ticket_id": ticket_id,
            "reply": reply,
            "sent_at": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        replies_path.write_text(json.dumps(replies, indent=4, ensure_ascii=False), encoding="utf-8")

    def latest_by_ticket(self) -> dict[str, dict]:
        """Latest correction per ticket id (later corrections win)."""
        return {str(c["ticket_id"]): c for c in self.corrections}

    def calculate_kpi(self, total_processed: int, ticket_ids=None) -> dict:
        """Calculates business performance metrics (UAT KPIs). Pass ticket_ids to ignore corrections for tickets no longer in the data."""
        corrected = self.latest_by_ticket()  # re-corrections of a ticket count once
        if ticket_ids is not None:
            known = {str(t) for t in ticket_ids}
            corrected = {t: c for t, c in corrected.items() if t in known}
        total_corrections = len(corrected)
        accuracy = 100.0
        if total_processed > 0:
            accuracy = ((total_processed - total_corrections) / total_processed) * 100.0

        return {
            "total_tickets_processed": total_processed,
            "human_interventions": total_corrections,
            "routing_accuracy": f"{round(accuracy, 2)}%"
        }