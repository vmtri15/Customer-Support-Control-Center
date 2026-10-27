import streamlit as st

st.set_page_config(page_title="Customer Support Platform", layout="wide", page_icon="🎧")

page = st.navigation([
    st.Page("app_pages/agent_console.py", title="Agent Console", icon="🎧", default=True),
    st.Page("app_pages/analytics.py", title="Analytics", icon="📊"),
    st.Page("app_pages/admin.py", title="Admin", icon="⚙️"),
])
page.run()
