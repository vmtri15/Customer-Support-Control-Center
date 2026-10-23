from modules.processor import TicketProcessor
from modules.feedback_manager import FeedbackManager

def run_human_in_the_loop_demo(max_reviews: int = 5):
    print("🚀 INITIATING TICKET ROUTING PROCESS (AI + HUMAN-IN-THE-LOOP)\n")

    # Initialize modules
    processor = TicketProcessor(file_path="data/tickets.csv")
    feedback_mgr = FeedbackManager(corrections_path="data/corrections.json")

    # 1. AI processes all tickets automatically
    print("🤖 Analyzing ticket data against company policies...")
    processed_tickets = processor.process_all()
    total_tickets = len(processed_tickets)
    flagged = [t for t in processed_tickets if t["routing_validation"]["requires_human_review"]]
    print(f"{len(flagged)} of {total_tickets} tickets need human review; showing the first {max_reviews}.")
    processed_tickets = flagged[:max_reviews]
    
    # 2. Human-in-the-loop Validation Step
    for ticket in processed_tickets:
        validation = ticket["routing_validation"]
        
        # If AI flags this ticket for human review (due to policy mismatch or low confidence)
        if validation["requires_human_review"]:
            print("-" * 60)
            print(f"⚠️ HUMAN REVIEW REQUIRED: Ticket {ticket['id']} from {ticket['customer']} (Tier: {ticket['tier']})")
            print(f"Content: '{ticket['email_content']}'")
            print(f"AI Recommended Routing: {validation['recommended_department']}")
            
            # Simulate manager providing feedback
            print("\n[Management Dashboard]")
            corrected_dept = input("Enter Correct Department (e.g., Retention, Logistics, Technical): ")
            reason = input("Reason for Override (e.g., VIP Escalation, Edge Case): ")
            
            # Save the human correction to knowledge base
            feedback_mgr.save_correction(
                ticket_id=ticket["id"],
                original_dept=validation["recommended_department"],
                corrected_dept=corrected_dept,
                reason=reason,
                email_content=ticket["email_content"],
            )
            print("-" * 60)
            
    # 3. Export Performance Report (UAT Metrics)
    print("\n📊 USER ACCEPTANCE TESTING (UAT) REPORT")
    kpi_report = feedback_mgr.calculate_kpi(total_processed=total_tickets)
    for key, value in kpi_report.items():
        # Formatting keys to look like standard business metrics
        formatted_key = key.replace('_', ' ').title()
        print(f"- {formatted_key}: {value}")

if __name__ == "__main__":
    run_human_in_the_loop_demo()