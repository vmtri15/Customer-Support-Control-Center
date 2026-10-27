"""Shared, cached objects for all Streamlit pages."""
import json

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
    agents = [a["name"] for a in store.list_agents()]
    return st.sidebar.selectbox("Acting as", agents, key="acting_agent")
