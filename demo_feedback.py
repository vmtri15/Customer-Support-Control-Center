from modules.processor import Orchestrator
from modules.feedback_manager import FeedbackManager

def run_human_in_the_loop_demo():
    print("🚀 INITIATING TICKET ROUTING PROCESS (AI + HUMAN-IN-THE-LOOP)\n")
    
    # Initialize modules
    processor = Orchestrator(config_path="workflow_config.json")
    feedback_mgr = FeedbackManager(corrections_path="data/corrections.json")
    
    # 1. AI processes all tickets automatically
    print("🤖 Analyzing ticket data against company policies...")
    processed_ticket = orchestrator.process_ticket(ticket_data)
    total_tickets = len(processed_tickets)
    
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
                reason=reason
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