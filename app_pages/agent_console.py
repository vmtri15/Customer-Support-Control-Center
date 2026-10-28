import datetime

import pandas as pd
import streamlit as st

from app_pages import common
from modules import store
from modules.email_sender import is_valid_address, send_email, smtp_configured, split_subject

common.ensure_store()
agent = common.current_agent()
processor = common.processor()
feedback = common.get_feedback()
departments = list(processor.policy.get("department_rules", {}))

AVATARS = {"customer": "👤", "agent": "🎧", "ai": "🤖", "system": "ℹ️"}
PRIORITY_BADGE = {"High": ":red-badge[High]", "Medium": ":orange-badge[Medium]", "Low": ":gray-badge[Low]"}

st.title("Agent Console")

# --- Overview of the open workload ---
open_tickets = store.list_tickets(statuses=store.OPEN_STATUSES)
sla = {t["id"]: store.sla(t) for t in open_tickets}
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Open tickets", len(open_tickets))
m2.metric("Needs review", sum(t["needs_review"] and not t["human_reviewed"] for t in open_tickets))
m3.metric("Unassigned", sum(t["assignee"] is None for t in open_tickets))
m4.metric("High priority", sum(t["priority"] == "High" for t in open_tickets))
m5.metric("Overdue", sum(s["overdue"] for s in sla.values()), help="Past the first-reply or resolution target "
          + ", ".join(f"({p}: {a} h / {b} h)" for p, (a, b) in store.SLA_HOURS.items()))

# --- Queue filters ---
f1, f2, f3, f4 = st.columns([1.2, 2, 1, 1.5])
dept = f1.selectbox("Department", ["All"] + departments)
statuses = f2.multiselect("Status", store.STATUSES, default=store.OPEN_STATUSES)
who = f3.selectbox("Assignee", ["Anyone", "Me", "Unassigned"])
search = f4.text_input("Search", placeholder="ID, customer or text")
t1, t2, _ = st.columns([1, 1, 3])
review_only = t1.toggle("Needs review only")
overdue_only = t2.toggle("Overdue only")

queue = store.list_tickets(
    statuses=statuses or None,
    category=None if dept == "All" else dept,
    assignee={"Anyone": None, "Me": agent, "Unassigned": ""}[who],
    needs_review=True if review_only else None,
    search=search or None,
)
if overdue_only:
    queue = [t for t in queue if sla.get(t["id"], {}).get("overdue")]


def due_text(ticket_id: str) -> str:
    s = sla.get(ticket_id)
    if not s or s["next_due"] is None:
        return "—"
    hours = (s["next_due"] - datetime.datetime.now()).total_seconds() / 3600
    return f"⏰ {-hours:.1f} h overdue" if hours < 0 else f"{hours:.1f} h ({s['next_target']})"


if not queue:
    st.info("No tickets match these filters.")
    st.stop()

queue_df = pd.DataFrame([{
    "ID": t["id"], "Priority": t["priority"], "Department": t["category"],
    "Confidence": t["confidence"], "Status": t["status"], "Due": due_text(t["id"]), "Assignee": t["assignee"] or "—",
    "Customer": t["customer_name"], "Message": t["content"][:90], "Created": t["created_at"].replace("T", " "),
} for t in queue])
st.caption(f"{len(queue)} tickets · highest priority first, then oldest")
event = st.dataframe(
    queue_df, hide_index=True, height=280, on_select="rerun", selection_mode="single-row", key="queue",
    column_config={"Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)},
)
if event.selection.rows:
    st.session_state.selected_ticket = queue_df.iloc[event.selection.rows[0]]["ID"]
ticket_id = st.session_state.get("selected_ticket")
if ticket_id is None or store.get_ticket(ticket_id) is None:
    ticket_id = queue[0]["id"]
ticket = store.get_ticket(ticket_id)

st.divider()


# --- Reply composer callbacks (run before the rerun, so they may reset widget state) ---
def send_reply(tid: str, reply_key: str, to_key: str, wait_key: str) -> None:
    reply = st.session_state[reply_key].strip()
    to_addr = st.session_state[to_key].strip()
    if not reply:
        st.session_state.flash = ("error", "Write a reply first.")
        return
    subject, body = split_subject(reply, tid)
    emailed = False
    if smtp_configured() and to_addr:
        if not is_valid_address(to_addr):
            st.session_state.flash = ("error", f"Invalid email address: {to_addr}")
            return
        try:
            send_email(to_addr, subject, body)
            emailed = True
        except Exception as e:
            st.session_state.flash = ("error", f"Email not sent, reply kept: {e}")
            return
    store.add_message(tid, "agent", agent, body)
    if st.session_state[wait_key]:
        store.update_ticket(tid, agent, status="Waiting on Customer")
    st.session_state[reply_key] = ""
    st.session_state.flash = ("success", f"Reply sent to {to_addr}." if emailed else "Reply added to the thread (not emailed).")


def draft_with_ai(t: dict, reply_key: str) -> None:
    draft_input = {"id": t["id"], "category": t["category"], "customer": t["customer_name"] or "Customer",
                   "email_content": t["content"], "sentiment": t["sentiment"], "priority": t["priority"]}
    try:
        st.session_state[reply_key] = processor.generate_email_draft(draft_input)
    except Exception as e:
        st.session_state.flash = ("error", f"AI draft failed: {e}")


def save_department(t: dict, dept_key: str, reason_key: str) -> None:
    new_dept = st.session_state[dept_key]
    if new_dept != t["category"]:
        feedback.save_correction(t["id"], t["category"], new_dept, st.session_state[reason_key], t["content"])
    store.update_ticket(t["id"], agent, category=new_dept, human_reviewed=1, needs_review=0)
    st.session_state.flash = ("success", "Department corrected; the model will learn it on the next retrain."
                              if new_dept != t["category"] else "AI routing confirmed.")


# --- Ticket detail ---
flash = st.session_state.pop("flash", None)
if flash:
    getattr(st, flash[0])(flash[1])

st.subheader(f"{ticket['id']} · {ticket['customer_name'] or 'Customer'}")
if ticket["subject"]:
    st.markdown(f"**{ticket['subject']}**")
st.markdown(f"{PRIORITY_BADGE.get(ticket['priority'], '')} :blue-badge[{ticket['category']}] "
            f":violet-badge[{ticket['status']}] :gray-badge[{ticket['sentiment']}]")
if ticket["human_reviewed"]:
    st.success("Human reviewed: routing confirmed by an agent.")
elif ticket["needs_review"]:
    st.warning("Needs review: low AI confidence or the keywords point to another department.")
if ticket["routed_by_correction"]:
    match = ticket["routed_by_correction"]
    st.info(f"Routed like human-corrected ticket {match['ticket_id']} ({match['similarity']:.0%} similar).")
ticket_sla = store.sla(ticket)
if ticket_sla["overdue"]:
    late_h = (datetime.datetime.now() - ticket_sla["next_due"]).total_seconds() / 3600
    target_h = store.SLA_HOURS.get(ticket["priority"], store.SLA_HOURS["Medium"])[ticket_sla["next_target"] == "resolution"]
    st.error(f"⏰ {ticket_sla['next_target'].capitalize()} overdue by {late_h:.1f} h "
             f"(target {target_h} h for {ticket['priority']} priority).")
AI_OUTCOME_TEXT = {"pending": ("info", "🤖 AI answered; waiting for the customer to confirm it solved the problem."),
                   "solved": ("success", "🤖 Solved by the AI: the customer confirmed the answer."),
                   "escalated": ("warning", "🤖 The AI answer didn't solve it; the customer needs an agent.")}
if ticket["ai_outcome"]:
    getattr(st, AI_OUTCOME_TEXT[ticket["ai_outcome"]][0])(AI_OUTCOME_TEXT[ticket["ai_outcome"]][1])

thread_col, side_col = st.columns([3, 2], gap="large")

with thread_col:
    st.markdown("**Conversation**")
    for m in store.get_messages(ticket_id):
        with st.chat_message(m["author_type"], avatar=AVATARS.get(m["author_type"])):
            st.caption(f"{m['author'] or m['author_type'].title()} · {m['created_at'].replace('T', ' ')}")
            st.write(m["body"])

    reply_key, to_key, wait_key = f"reply_{ticket_id}", f"to_{ticket_id}", f"wait_{ticket_id}"
    st.button("✨ Draft reply with AI", key=f"ai_{ticket_id}", on_click=draft_with_ai, args=(ticket, reply_key))
    st.text_area("Reply", key=reply_key, height=200, placeholder="Write a reply, or draft one with AI and edit it")
    c1, c2 = st.columns(2)
    c1.text_input("Email to", value=ticket["customer_email"] or "", key=to_key,
                  help="Emailed only when SMTP is configured; otherwise the reply is just added to the thread.")
    c2.checkbox("Set status to Waiting on Customer", value=True, key=wait_key)
    st.button("Send reply" if smtp_configured() else "Add reply to thread", type="primary", key=f"send_{ticket_id}",
              on_click=send_reply, args=(ticket_id, reply_key, to_key, wait_key))

with side_col:
    with st.form(f"props_{ticket_id}"):
        st.markdown("**Ticket**")
        agents = [a["name"] for a in store.list_agents()]
        status = st.selectbox("Status", store.STATUSES, index=store.STATUSES.index(ticket["status"]))
        assignee = st.selectbox("Assignee", ["—"] + agents,
                                index=(agents.index(ticket["assignee"]) + 1) if ticket["assignee"] in agents else 0)
        priority = st.selectbox("Priority", store.PRIORITIES, index=store.PRIORITIES.index(ticket["priority"])
                                if ticket["priority"] in store.PRIORITIES else 1)
        if st.form_submit_button("Update ticket"):
            changes = store.update_ticket(ticket_id, agent, status=status, priority=priority,
                                          assignee=None if assignee == "—" else assignee)
            st.session_state.flash = ("success", "Updated: " + "; ".join(changes)) if changes else ("info", "No changes.")
            st.rerun()
    if ticket["assignee"] != agent and st.button("Assign to me", key=f"me_{ticket_id}"):
        store.update_ticket(ticket_id, agent, assignee=agent, status="In Progress" if ticket["status"] == "Open" else ticket["status"])
        st.rerun()

    st.markdown("**Routing**")
    dept_key, reason_key = f"dept_{ticket_id}", f"reason_{ticket_id}"
    st.selectbox("Department", departments, key=dept_key,
                 index=departments.index(ticket["category"]) if ticket["category"] in departments else None)
    st.text_input("Reason (if changing)", key=reason_key)
    st.button("Confirm / correct department", key=f"route_{ticket_id}",
              on_click=save_department, args=(ticket, dept_key, reason_key))

    st.markdown("**Details**")
    st.write(f"AI confidence: **{(ticket['confidence'] or 0):.0%}** · Tier: {ticket['tier'] or '—'} · Channel: {ticket['channel']}")
    st.write(f"Created {ticket['created_at'].replace('T', ' ')}")
    if ticket["first_response_at"]:
        st.write(f"First response {ticket['first_response_at'].replace('T', ' ')}")
    if ticket["resolved_at"]:
        st.write(f"Resolved {ticket['resolved_at'].replace('T', ' ')}")

    if ticket["probabilities"]:
        with st.expander("Department probabilities"):
            for cat, prob in sorted(ticket["probabilities"].items(), key=lambda x: x[1], reverse=True):
                st.write(f"- **{cat}**: {prob:.1%}")

    with st.expander("Similar past tickets"):
        if st.button("Search", key=f"similar_{ticket_id}"):
            try:
                from modules.vector_store import find_similar
                with st.spinner("Searching (the first search loads the index)..."):
                    hits = find_similar([ticket["content"]], n=6, exclude_ids=[ticket_id])[0]
                # Bỏ các vé trùng y hệt nội dung (dữ liệu cũ có vé lặp)
                for h in [h for h in hits if h["text"] != ticket["content"]][:5]:
                    source = "🧑 Human correction" if h["source"] == "human_correction" else "Dataset"
                    st.markdown(f"**{h['similarity']:.0%} similar** · {h['category']} · {source}  \n{h['text']}")
                    if h.get("response"):
                        st.caption(h["response"][:400] + ("…" if len(h["response"]) > 400 else ""))
            except Exception as e:
                st.error(f"Similar ticket search failed: {e}")

    with st.expander("Activity log"):
        for e in store.get_events(ticket_id):
            st.write(f"`{e['created_at'].replace('T', ' ')}` **{e['actor']}** · {e['type']}: {e['detail']}")
