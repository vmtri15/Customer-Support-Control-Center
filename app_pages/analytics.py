import pandas as pd
import plotly.express as px
import streamlit as st

from app_pages import common
from modules import store

common.ensure_store()
common.current_agent()
feedback = common.get_feedback()

st.title("Analytics")

df = pd.DataFrame(store.list_tickets())
if df.empty:
    st.info("No tickets yet.")
    st.stop()

is_open = df["status"].isin(store.OPEN_STATUSES)
responded = df.dropna(subset=["first_response_at"])
first_response_h = (pd.to_datetime(responded["first_response_at"]) - pd.to_datetime(responded["created_at"])).dt.total_seconds() / 3600
kpi = feedback.calculate_kpi(total_processed=len(df), ticket_ids=df["id"].tolist())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total tickets", len(df))
c2.metric("Open", int(is_open.sum()))
c3.metric("Resolved", int((df["status"] == "Resolved").sum()))
c4.metric("Human corrections", kpi["human_interventions"], help=f"Routing accuracy: {kpi['routing_accuracy']}")
c5.metric("Median first response", f"{first_response_h.median():.1f} h" if len(first_response_h) else "—")

st.divider()
col1, col2 = st.columns(2)
with col1:
    st.write("**Tickets by department**")
    fig = px.pie(df, names="category", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(hovertemplate="<b>%{label}</b><br>Tickets: %{value}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.write("**Status by department**")
    st.plotly_chart(px.histogram(df, y="category", color="status", barmode="stack",
                                 category_orders={"status": store.STATUSES}), use_container_width=True)

st.write("**AI confidence by department**")
st.plotly_chart(px.box(df, x="category", y="confidence", color="category"), use_container_width=True)
