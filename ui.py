import streamlit as st

st.set_page_config(page_title="Customer Support Platform", layout="wide", page_icon="🎧")

signed_in = bool(st.session_state.get("agent"))
help_center = st.Page("app_pages/customer_portal.py", title="Help Center", icon="💬", default=not signed_in)
if signed_in:
    team = [st.Page("app_pages/agent_console.py", title="Agent Console", icon="🎧", default=True),
            st.Page("app_pages/analytics.py", title="Analytics", icon="📊"),
            st.Page("app_pages/admin.py", title="Admin", icon="⚙️")]
else:  # customers only ever see the Help Center
    team = [st.Page("app_pages/sign_in.py", title="Agent sign in", icon="🔐")]
st.navigation({"Customers": [help_center], "Support team": team}).run()
