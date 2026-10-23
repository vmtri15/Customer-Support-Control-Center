"""Self-check for TicketProcessor.validate_routing. Run: python test_routing.py"""
import json

from modules.processor import TicketProcessor

# Skip __init__ so the ML model isn't loaded/trained
p = TicketProcessor.__new__(TicketProcessor)
with open("data/company_policy.json", encoding="utf-8") as f:
    p.policy = json.load(f)


def check(category, content, confidence=0.9):
    ticket = {"category": category, "email_content": content, "confidence_score": confidence}
    ticket["tags"] = p.assign_tags(content)
    return p.validate_routing(ticket)


# ML and keywords agree -> routed on the ML category, no review
r = check("Billing", "I want a refund for my invoice")
assert r["recommended_department"] == "Billing" and not r["requires_human_review"], r

# ML and keywords disagree -> review
r = check("Technical", "I want a refund for my invoice")
assert r["requires_human_review"] and r["recommended_department"] == "Technical", r

# No keyword evidence is not a conflict
r = check("Technical", "hello there")
assert not r["requires_human_review"], r

# Low confidence -> review even when keywords agree
r = check("Billing", "refund please", confidence=0.3)
assert r["requires_human_review"] and r["low_confidence"], r

# Unknown category -> review, no department
r = check("Nonsense", "refund please")
assert r["requires_human_review"] and r["recommended_department"] is None, r

print("✅ validate_routing checks passed")
