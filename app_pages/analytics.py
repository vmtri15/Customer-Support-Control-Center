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
created = pd.to_datetime(df["created_at"])
first_reply_h = (pd.to_datetime(df["first_response_at"]) - created).dt.total_seconds() / 3600
resolution_h = (pd.to_datetime(df["resolved_at"]) - created).dt.total_seconds() / 3600
sla = pd.DataFrame([store.sla(t) for t in df.to_dict("records")])
replied, resolved = df["first_response_at"].notna(), df["resolved_at"].notna()
kpi = feedback.calculate_kpi(total_processed=len(df), ticket_ids=df["id"].tolist())
ai = df["ai_outcome"].value_counts()
ai_decided = ai.get("solved", 0) + ai.get("escalated", 0)


def pct(x) -> str:
    return f"{x:.0%}" if pd.notna(x) else "—"


def hours(x) -> str:
    return f"{x:.1f} h" if pd.notna(x) else "—"


targets = ", ".join(f"{p}: {a} h / {b} h" for p, (a, b) in store.SLA_HOURS.items())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total tickets", len(df))
c2.metric("Open", int(is_open.sum()))
c3.metric("Overdue", int((sla["overdue"] & is_open).sum()), help=f"Targets (first reply / resolution): {targets}")
c4.metric("Corrections", kpi["human_interventions"], help=f"Routing accuracy: {kpi['routing_accuracy']}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Median first reply", hours(first_reply_h.median()),
          help=f"First reply on time: {pct(1 - sla.loc[replied, 'first_response_breached'].mean())}")
c2.metric("Median resolution", hours(resolution_h.median()),
          help=f"Resolved on time: {pct(1 - sla.loc[resolved, 'resolution_breached'].mean())}")
c3.metric("Satisfaction", f"{df['csat'].mean():.1f} / 5" if df["csat"].notna().any() else "—",
          help=f"{int(df['csat'].notna().sum())} ratings")
c4.metric("AI solved", pct(ai.get("solved", 0) / ai_decided) if ai_decided else "—",
          help=f"Of AI-answered tickets the customer judged: {ai.get('solved', 0)} solved, "
               f"{ai.get('escalated', 0)} needed an agent, {ai.get('pending', 0)} awaiting the customer.")

st.write("**Daily volume**")
daily = pd.DataFrame({
    "Created": created.dt.floor("D").value_counts(),
    "Resolved": pd.to_datetime(df["resolved_at"]).dt.floor("D").value_counts(),
}).fillna(0).sort_index()
st.line_chart(daily)

st.write("**Response times by priority**")
on_time = pd.DataFrame({
    "Priority": df["priority"], "Median first reply (h)": first_reply_h, "Median resolution (h)": resolution_h,
    "First reply on time": (1 - sla["first_response_breached"].astype(float)).where(replied),
    "Resolved on time": (1 - sla["resolution_breached"].astype(float)).where(resolved),
    "Satisfaction": df["csat"],
}).groupby("Priority").agg({"Median first reply (h)": "median", "Median resolution (h)": "median",
                            "First reply on time": "mean", "Resolved on time": "mean", "Satisfaction": "mean"})
st.dataframe(on_time.reindex([p for p in store.PRIORITIES if p in on_time.index]), column_config={
    c: st.column_config.NumberColumn(format="percent") for c in ["First reply on time", "Resolved on time"]
} | {c: st.column_config.NumberColumn(format="%.1f") for c in
     ["Median first reply (h)", "Median resolution (h)", "Satisfaction"]})

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
