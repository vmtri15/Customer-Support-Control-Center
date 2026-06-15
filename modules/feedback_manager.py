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
            "status": "Human_Reviewed"
        }
        self.corrections.append(correction)
        
        with self.corrections_path.open("w", encoding="utf-8") as f:
            json.dump(self.corrections, f, indent=4, ensure_ascii=False)
            
        # Update Vector DB to complete the Learning Loop
        if email_content:
            try:
                from modules.vector_store import get_chroma_client, get_ticket_collection
                client = get_chroma_client()
                collection = get_ticket_collection(client)
                
                collection.add(
                    documents=[email_content],
                    metadatas=[{"category": corrected_dept, "source": "human_correction"}],
                    ids=[f"correction_{ticket_id}"]
                )
                print(f"🧠 Vector DB updated: AI learned that Ticket {ticket_id} belongs to {corrected_dept}")
            except Exception as e:
                print(f"⚠️ Failed to update Vector DB: {e}")

        print(f"✅ Feedback logged for Ticket {ticket_id}: Successfully re-routed to {corrected_dept}")

    def calculate_kpi(self, total_processed: int) -> dict:
        """Calculates business performance metrics (UAT KPIs)."""
        total_corrections = len(self.corrections)
        accuracy = 100.0
        if total_processed > 0:
            accuracy = ((total_processed - total_corrections) / total_processed) * 100.0

        return {
            "total_tickets_processed": total_processed,
            "human_interventions": total_corrections,
            "routing_accuracy": f"{round(accuracy, 2)}%"
        }