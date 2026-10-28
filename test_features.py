"""
Self-checks for training, feedback, batch prediction and the Vector DB features.
Runs in a temporary folder, so your real data/models/chroma_db are untouched.
Run: python test_features.py   (~30s: trains a tiny model and loads the embedding model)
"""
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parent
os.chdir(tempfile.mkdtemp())
Path("data").mkdir()
shutil.copy(ROOT / "data/company_policy.json", "data/")

BILLING = ["Please refund my invoice payment", "I was charged twice on my card", "Where is my invoice for last month",
           "The fee on my bill is wrong", "I need a copy of my receipt", "Why was my payment declined"]
LOGISTICS = ["My package has not arrived yet", "Where is my delivery", "The shipping is very late",
             "Tracking shows no update for days", "Can I change my delivery address", "The courier lost my parcel"]
rows = [{"id": f"T{i}", "email_content": t, "category": "Billing", "response": f"Billing reply {i}"} for i, t in enumerate(BILLING)]
rows += [{"id": f"T{i + 6}", "email_content": t, "category": "Logistics", "response": ""} for i, t in enumerate(LOGISTICS)]
rows.append({"id": "T12", "email_content": "The app crashes", "category": "Technical", "response": ""})  # 1 vé: phải bị bỏ qua
pd.DataFrame(rows).to_csv("data/tickets.csv", index=False)
source_bytes = Path("data/tickets.csv").read_bytes()

# T0 được nhân viên sửa thành Logistics; NEW là vé không có trong file
json.dump([
    {"ticket_id": "T0", "corrected_department": "Billing", "email_content": BILLING[0]},
    {"ticket_id": "T0", "corrected_department": "Logistics", "email_content": BILLING[0]},
    {"ticket_id": "NEW", "corrected_department": "Billing", "email_content": "Double charge on my statement"},
], open("data/corrections.json", "w"))

from modules.feedback_manager import FeedbackManager  # noqa: E402
from modules.processor import (  # noqa: E402
    MLClassifier, TicketProcessor, _training_lock, llm_label, train_new_model_background,
)

# --- FeedbackManager ---
fm = FeedbackManager("data/corrections.json")
assert fm.latest_by_ticket()["T0"]["corrected_department"] == "Logistics", "later correction must win"
assert fm.calculate_kpi(10)["human_interventions"] == 2, "re-corrections count once"
assert fm.calculate_kpi(10, ticket_ids=["T0", "T1"])["human_interventions"] == 1, "unknown tickets ignored"
print("✅ FeedbackManager")

# --- Training ---
m = MLClassifier("data/tickets.csv", force_retrain=True)
assert m.is_trained and set(m.model.classes_) == {"Billing", "Logistics"}, m.model.classes_
assert Path("data/tickets.csv").read_bytes() == source_bytes, "training must not modify the source file"
df = m._apply_corrections(pd.read_csv("data/tickets.csv"))
assert df.loc[df["id"] == "T0", "category"].item() == "Logistics" and "NEW" in set(df["id"])
meta = json.load(open("models/metadata.json"))
assert meta["active_version"] in meta["versions"]
print("✅ Training")

# --- Only one background training at a time ---
versions_before = len(json.load(open("models/metadata.json"))["versions"])
with _training_lock:
    train_new_model_background("data/tickets.csv")  # phải bỏ qua vì đang có lần train khác
assert len(json.load(open("models/metadata.json"))["versions"]) == versions_before
print("✅ Training lock")

# --- Batch prediction matches single prediction ---
texts = ["my parcel is late", "refund my payment", "", "   "]
batch = m.predict_many(texts)
for t, b in zip(texts, batch):
    assert b == m.predict(t), (t, b)
assert batch[2] == ("General Inquiry", 0.0, {})
print("✅ predict_many")

# --- LLM label validation ---
for answer, want in [("Billing.", "Billing"), ("billing or logistics", None), ("Refunds", None)]:
    with patch("modules.llm_client.chat_completion", return_value=answer):
        assert llm_label("x", ["Billing", "Logistics"]) == want, answer
print("✅ llm_label")

# --- Vector DB: similar tickets, RAG prompt, corrections generalize ---
from modules.vector_store import find_similar  # noqa: E402

hits = find_similar([BILLING[1]], n=3, exclude_ids=["T1"])[0]
assert hits and all(h["id"] not in ("T1", "correction_T1") for h in hits), hits

p = TicketProcessor("data/tickets.csv")
ticket = {"id": "T1", "email_content": BILLING[1], "category": "Billing", "customer": "A"}
with patch("modules.llm_client.chat_completion", return_value="draft") as llm:
    p.generate_email_draft(ticket)
prompt = llm.call_args.args[0]
assert "Nhân viên:" in prompt and "Billing reply" in prompt, prompt
print("✅ find_similar + RAG prompt")

# --- AI solver: grounded in sample replies, hands over on ESCALATE ---
from modules.processor import auto_solvable  # noqa: E402

sure = {"confidence": 0.9, "needs_review": 0, "priority": "Medium", "sentiment": "Neutral"}
assert auto_solvable(sure)
for unsure in [{"confidence": 0.7}, {"needs_review": 1}, {"priority": "High"}, {"sentiment": "Negative"}]:
    assert not auto_solvable({**sure, **unsure}), unsure
with patch("modules.llm_client.chat_completion", return_value="Hi A, here is how...") as llm:
    assert p.solve_ticket(ticket) == "Hi A, here is how..."
    assert "Billing reply" in llm.call_args.args[0]
with patch("modules.llm_client.chat_completion", return_value="ESCALATE"):
    assert p.solve_ticket(ticket) is None
with patch.object(p, "reply_examples", return_value=""), patch("modules.llm_client.chat_completion") as llm:
    assert p.solve_ticket(ticket) is None and not llm.called  # no grounding, no LLM call
print("✅ AI solver")

# A near-copy of T0's text with low confidence must follow T0's correction (Logistics)
p.tickets = [{"id": "X1", "email_content": "Please refund my invoice payment now"},
             {"id": "X2", "email_content": "Totally unrelated question about store hours"}]
with patch.object(p.classifier, "predict_many", return_value=[("Billing", 0.3, {}), ("Billing", 0.3, {})]):
    out = p.process_all()
assert out[0]["category"] == "Logistics" and out[0]["routed_by_correction"]["ticket_id"] == "T0", out[0]
assert out[1]["category"] == "Billing" and "routed_by_correction" not in out[1]
print("✅ low-confidence tickets follow similar corrections")

# --- Email sending (SMTP mocked: nothing is really sent) ---
from modules import email_sender  # noqa: E402

assert email_sender.split_subject("SUBJECT: Update - Ref: T1\n\nHi A,\nBody", "T1") == ("Update - Ref: T1", "Hi A,\nBody")
assert email_sender.split_subject("Hi A, no subject line", "T1")[0] == "Re: your support request - Ref: T1"
for bad in ["", "not-an-email", "a@b", "x@y.com\nBcc: evil@z.com"]:
    assert not email_sender.is_valid_address(bad), bad

env = {"SMTP_HOST": "smtp.example.com", "SMTP_PORT": "587", "SMTP_FROM": "support@example.com",
       "SMTP_USER": "support@example.com", "SMTP_PASSWORD": "test-only"}
with patch.dict(os.environ, env, clear=False), patch("smtplib.SMTP") as smtp:
    assert email_sender.smtp_configured()
    email_sender.send_email("customer@example.com", "Update - Ref: T1", "Hi A,\nBody")
    server = smtp.return_value
    server.starttls.assert_called_once()
    server.login.assert_called_once_with("support@example.com", "test-only")
    msg = server.send_message.call_args.args[0]
    assert msg["To"] == "customer@example.com" and msg["Subject"] == "Update - Ref: T1"
    assert "Body" in msg.get_content()
    try:
        email_sender.send_email("not-an-email", "s", "b")
        raise AssertionError("invalid address must be rejected")
    except ValueError:
        pass
    assert server.send_message.call_count == 1
with patch.dict(os.environ, {**env, "SMTP_PORT": "465"}), patch("smtplib.SMTP_SSL") as smtp_ssl:
    email_sender.send_email("customer@example.com", "s", "b")
    smtp_ssl.return_value.send_message.assert_called_once()
    smtp_ssl.return_value.starttls.assert_not_called()
print("✅ email sending")
print("✅ all feature checks passed")
