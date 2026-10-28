import streamlit as st

from app_pages import common
from modules import store
from modules.email_sender import is_valid_address, split_subject
from modules.processor import auto_solvable

common.ensure_store()

AVATARS = {"customer": "👤", "agent": "🎧", "ai": "🤖"}
STATUS_TEXT = {
    "Open": "We've received your request and will reply soon.",
    "In Progress": "An agent is working on your request.",
    "Waiting on Customer": "We've replied and are waiting for your answer.",
    "Resolved": "This request is resolved. Reply below if you still need help.",
}

st.title("Help Center")

flash = st.session_state.pop("portal_flash", None)
if flash:
    getattr(st, flash[0])(flash[1])

# A radio (unlike st.tabs) keeps its selection across reruns and, unlike st.segmented_control, cannot be deselected
VIEWS = ["Submit a request", "Track my request"]
if "portal_goto" in st.session_state:  # set before a rerun; widget state can't change once rendered
    st.session_state.portal_view = st.session_state.pop("portal_goto")
st.session_state.setdefault("portal_view", VIEWS[0])
view = st.radio("View", VIEWS, key="portal_view", horizontal=True, label_visibility="collapsed")

if view != VIEWS[1]:
    with st.form("new_request", clear_on_submit=True):
        c1, c2 = st.columns(2)
        name = c1.text_input("Your name")
        email = c2.text_input("Email")
        subject = st.text_input("Subject")
        content = st.text_area("How can we help?", height=150, max_chars=common.MAX_MESSAGE_CHARS)
        submitted = st.form_submit_button("Submit request", type="primary")
    if submitted:
        if not (name.strip() and content.strip()):
            st.error("Please fill in your name and message.")
        elif not is_valid_address(email.strip()):
            st.error("Please enter a valid email address, so you can track your request.")
        elif (store.web_tickets_since(1, email) >= common.TICKETS_PER_EMAIL_HOURLY
              or store.web_tickets_since(1) >= common.TICKETS_HOURLY):
            st.error("We've received a lot of requests in the last hour. Please try again later.")
        else:
            with st.spinner("Sending your request..."):
                triage = common.triage(common.processor(), "new", content, name.strip())  # store assigns the id
                ticket_id = store.create_ticket(name.strip(), email.strip(), subject.strip() or None, content,
                                                triage, channel="web")
            if auto_solvable(triage) and common.llm_allowed():
                with st.spinner("Looking for an answer..."):
                    try:
                        store.log_llm_call(ticket_id, "AI", "solver")
                        answer = common.processor().solve_ticket(
                            {"id": ticket_id, "email_content": content, "customer": name.strip(),
                             "category": triage["category"]})
                    except Exception as e:  # LLM down: the ticket simply waits for an agent
                        print(f"⚠️ AI solver failed for {ticket_id}: {e}")
                        answer = None
                if answer:
                    store.ai_answer(ticket_id, split_subject(answer, ticket_id)[1])
            st.session_state.portal_auth = (ticket_id, email.strip())
            st.session_state.portal_goto = VIEWS[1]
            st.session_state.portal_flash = (
                "success", f"Request received. Your ticket number is **{ticket_id}**. "
                           "Keep it with your email to check on the request later.")
            st.rerun()

else:
    auth = st.session_state.get("portal_auth")
    if auth is None:
        with st.form("lookup"):
            c1, c2 = st.columns(2)
            lookup_id = c1.text_input("Ticket number", placeholder="TK1A2B3C4D")
            lookup_email = c2.text_input("Email used for the request")
            if st.form_submit_button("Find my request"):
                found = store.get_customer_ticket(lookup_id, lookup_email)
                if found:
                    st.session_state.portal_auth = (found["id"], lookup_email.strip())
                    st.rerun()
                st.error("No request found with that ticket number and email.")
    else:
        ticket = store.get_customer_ticket(*auth)
        if ticket is None:  # deleted or reset demo database
            st.session_state.pop("portal_auth")
            st.rerun()
        st.subheader(ticket["subject"] or "Your request")
        st.caption(f"Ticket {ticket['id']} · opened {ticket['created_at'].replace('T', ' ')} · {ticket['category']}")
        st.info(f"**{ticket['status']}** — {STATUS_TEXT[ticket['status']]}")

        for m in store.get_messages(ticket["id"]):
            if m["author_type"] not in AVATARS:  # system notes are internal
                continue
            role = "user" if m["author_type"] == "customer" else "assistant"
            with st.chat_message(role, avatar=AVATARS[m["author_type"]]):
                who = "You" if m["author_type"] == "customer" else ("Support AI" if m["author_type"] == "ai" else m["author"])
                st.caption(f"{who} · {m['created_at'].replace('T', ' ')}")
                st.markdown(m["body"])

        if ticket["ai_outcome"] == "pending":
            st.write("**Did this answer solve your problem?**")
            b1, b2, _ = st.columns([1, 1, 3])
            if b1.button("✅ Yes, this solved it", type="primary"):
                store.customer_verdict(ticket["id"], solved=True)
                st.session_state.portal_flash = ("success", "Glad we could help! Your request is now resolved.")
                st.rerun()
            if b2.button("🙋 I still need help"):
                store.customer_verdict(ticket["id"], solved=False)
                st.session_state.portal_flash = ("info", "Thanks, a support agent will take over your request.")
                st.rerun()

        if ticket["status"] == "Resolved" and ticket["csat"] is None:
            st.write("**How did we do?**")
            rating = st.feedback("stars", key=f"csat_{ticket['id']}")
            if rating is not None:  # 0-4 index
                store.rate_ticket(ticket["id"], rating + 1)
                st.session_state.portal_flash = ("success", "Thanks for your feedback!")
                st.rerun()

        with st.form("customer_reply", clear_on_submit=True):
            reply = st.text_area("Reply", height=100, max_chars=common.MAX_MESSAGE_CHARS)
            if st.form_submit_button("Send reply", type="primary"):
                if reply.strip():
                    store.customer_reply(ticket["id"], reply)
                    st.session_state.portal_flash = ("success", "Reply sent.")
                    st.rerun()
                st.error("Write a message first.")
        if st.button("Look up a different request"):
            st.session_state.pop("portal_auth")
            st.rerun()
