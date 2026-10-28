import hmac
import os
import time

import streamlit as st

from app_pages import common
from modules import store

common.ensure_store()
password = os.getenv("AGENT_PASSWORD")

st.title("Agent sign in")
if not password:
    st.info("Demo mode: pick any agent to look around. Set `AGENT_PASSWORD` to require a password.")
with st.form("sign_in"):
    agent = st.selectbox("Agent", [a["name"] for a in store.list_agents()])
    typed = st.text_input("Password", type="password") if password else ""
    if st.form_submit_button("Sign in", type="primary"):
        if password and not hmac.compare_digest(typed.encode(), password.encode()):
            time.sleep(1)  # slows down password guessing
            st.error("Wrong password.")
        else:
            st.session_state.agent = agent
            st.rerun()
