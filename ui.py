import streamlit as st
import pandas as pd
# Import plotly để vẽ biểu đồ
import plotly.express as px 
from modules.processor import TicketProcessor
from modules.feedback_manager import FeedbackManager

st.set_page_config(page_title="Enterprise AI Ticket Router", layout="wide", page_icon="🚀")

st.title("Customer Support Control Center")

# Load hệ thống
@st.cache_resource
def init_system():
    p = TicketProcessor(file_path="data/tickets.csv")
    f = FeedbackManager(corrections_path="data/corrections.json")
    return p, f

@st.cache_data
def get_processed_tickets(_processor):
    return _processor.process_all()

processor, feedback_mgr = init_system()
tickets = get_processed_tickets(processor)
df_tickets = pd.DataFrame(tickets) # Chuyển thành DataFrame để vẽ biểu đồ dễ hơn

# --- TẠO 2 TABS CHUYÊN NGHIỆP ---
tab_management, tab_analytics = st.tabs(["Triage & Management", "Analytics Hub"])

# ==========================================
# TAB 1: TICKET MANAGEMENT 
# ==========================================
with tab_management:
    st.sidebar.header("Ticket Explorer")
    ticket_ids = df_tickets['id'].tolist()
    selected_id = st.sidebar.selectbox("Select a Ticket to Inspect:", ticket_ids)

    # Lấy ticket đang chọn
    ticket = next(t for t in tickets if t['id'] == selected_id)
    confidence = float(ticket.get('confidence_score', 0))

    st.subheader(f"Ticket Detail: {ticket['id']}")
    
    # Warning if AI is not confident
    if confidence < 0.65:
        st.error("Human Review Required: AI Confidence Score is below threshold (65%).")

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Customer:** {ticket.get('customer', 'N/A')}")
        st.write(f"**Created at:** {ticket.get('timestamp', 'N/A')}")
        st.write(f"**Content:** {ticket['email_content']}")
    with col2:
        st.warning(f"**Status:** {ticket.get('status', 'N/A')}")
        
        if ticket.get('original_csv_category') and ticket['original_csv_category'] != ticket['category']:
            st.write(f"**Category:** {ticket.get('category', 'N/A')} *(Changed from {ticket['original_csv_category']})*")
        else:
            st.write(f"**Category:** {ticket.get('category', 'N/A')}")
            
        st.metric("Confidence Score", f"{confidence*100:.1f}%")
        
        # Show all probabilities in a clean dropdown
        if "category_probabilities" in ticket and ticket["category_probabilities"]:
            with st.expander("View All Category Probabilities"):
                sorted_probs = sorted(ticket["category_probabilities"].items(), key=lambda x: x[1], reverse=True)
                for cat, p in sorted_probs:
                    st.write(f"- **{cat}**: {p*100:.1f}%")

    st.divider()
    
    # --- New Feature: GenAI Auto-Reply ---
    st.subheader("Auto-Reply Draft")

    # Use session state to persist the draft content across reruns
    if "draft_content" not in st.session_state:
        st.session_state.draft_content = ""

    if st.button("Generate AI Reply", key="generate_draft"):
        st.session_state.draft_content = ""  # Clear previous draft
        try:
            with st.spinner("Generating response via LLM..."):
                stream = processor.generate_email_draft(ticket, stream=True)
                
                # Create a placeholder for the typewriter effect
                response_placeholder = st.empty()
                full_response = ""
                for chunk in stream:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌") # Add a cursor effect
                response_placeholder.markdown(full_response)
                
                # Once streaming is done, save the final draft to session state
                st.session_state.draft_content = full_response

        except Exception as e:
            st.error(f"Failed to generate AI response. Details: {e}")
            st.session_state.draft_content = "" # Clear on error

    # Display the editable text area if a draft exists
    if st.session_state.draft_content:
        st.text_area("Review and edit before sending:", value=st.session_state.draft_content, height=250, key="editable_draft")
        if st.button("Send to Customer", key="send_draft"):
            st.success("Email draft sent successfully!")
            st.session_state.draft_content = "" # Clear draft after sending
            st.rerun()

    # --- KHU VỰC HUMAN-IN-THE-LOOP ---
    st.subheader("Human Intervention Panel")
    with st.container():
        col_a, col_b = st.columns(2)
        with col_a:
            new_dept = st.text_input("Correct Department (If AI is wrong):")
            reason = st.text_area("Reason for Change:")
        with col_b:
            if st.button("Save Correction & Route"):
                feedback_mgr.save_correction(selected_id, ticket['priority'], new_dept, reason, ticket['email_content'])
                st.success("Ticket updated & AI has learned this correction!")

# ==========================================
# TAB 2: ANALYTICS HUB (Level 2: Data Visualization)
# ==========================================
with tab_analytics:
    st.subheader("System Performance & Traffic Analysis")
    
    # Tính KPI
    kpi = feedback_mgr.calculate_kpi(total_processed=len(tickets))
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("Total Tickets Processed", kpi['total_tickets_processed'])
    col_kpi2.metric("Human Interventions", kpi['human_interventions'])
    col_kpi3.metric("Routing Accuracy", kpi['routing_accuracy'])
    
    st.divider()
    
    # Vẽ biểu đồ
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.write("**Ticket Distribution by Category**")
        fig_pie = px.pie(df_tickets, names='category', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_traces(hovertemplate='<b>%{label}</b><br>Tickets: %{value}<extra></extra>') 
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        st.write("**AI Confidence Score vs Category**")
        fig_box = px.box(df_tickets, x="category", y="confidence_score", color="category")
        st.plotly_chart(fig_box, use_container_width=True)