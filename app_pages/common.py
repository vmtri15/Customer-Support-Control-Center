"""Shared, cached objects for all Streamlit pages."""
import json
import os

import streamlit as st

from modules import store
from modules.feedback_manager import FeedbackManager
from modules.processor import TicketProcessor


def active_model_version():
    try:
        with open("models/metadata.json", "r") as f:
            return json.load(f).get("active_version")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# Cache theo model_version để tự nạp lại model sau khi train xong
@st.cache_resource
def get_processor(model_version) -> TicketProcessor:
    return TicketProcessor(file_path="data/tickets.csv")


@st.cache_resource
def get_feedback() -> FeedbackManager:
    return FeedbackManager(corrections_path="data/corrections.json")


@st.cache_resource
def ensure_store() -> None:
    """Create the database and, on first run, fill it with the triaged demo tickets."""
    store.init_db()
    if store.ticket_count() == 0:
        with st.spinner("First run: triaging demo tickets into the database..."):
            store.seed_from_processed(processor().process_all())


def processor() -> TicketProcessor:
    return get_processor(active_model_version())


def triage(processor_: TicketProcessor, ticket_id: str, content: str, customer: str = "") -> dict:
    """Run the AI triage on one ticket and return store columns."""
    ticket = {"id": ticket_id, "email_content": content, "customer": customer}
    return store.triage_fields(processor_.process_many([ticket])[0])


def current_agent() -> str:
    agent = st.session_state["agent"]  # ui.py only shows support-team pages after sign-in
    st.sidebar.caption(f"Signed in as **{agent}**")
    if st.sidebar.button("Sign out"):
        del st.session_state["agent"]
        st.rerun()
    return agent


# Abuse limits for a public demo; counted in the database, so reloading the page doesn't reset them
LLM_DAILY_LIMIT = int(os.getenv("LLM_DAILY_LIMIT", "100"))
TICKETS_PER_EMAIL_HOURLY = int(os.getenv("TICKETS_PER_EMAIL_HOURLY", "3"))
TICKETS_HOURLY = int(os.getenv("TICKETS_HOURLY", "60"))
MAX_MESSAGE_CHARS = 4000


def llm_allowed() -> bool:
    return store.llm_calls_since(24) < LLM_DAILY_LIMIT
