import streamlit as st
import pandas as pd
import threading
import json
# Import plotly để vẽ biểu đồ
import plotly.express as px 
from modules.processor import TicketProcessor, CONFIDENCE_THRESHOLD
from modules.processor import train_new_model_background, is_training
from modules.feedback_manager import FeedbackManager

st.set_page_config(page_title="Enterprise AI Ticket Router", layout="wide", page_icon="🚀")

st.title("Customer Support Control Center")

def active_model_version():
    try:
        with open("models/metadata.json", "r") as f:
            return json.load(f).get("active_version")
    except (FileNotFoundError, json.JSONDecodeError):
        return None

# Load hệ thống; cache theo model_version để tự nạp lại model sau khi train xong
@st.cache_resource
def init_system(model_version):
    p = TicketProcessor(file_path="data/tickets.csv")
    f = FeedbackManager(corrections_path="data/corrections.json")
    return p, f

@st.cache_data
def get_processed_tickets(_processor, model_version):
    return _processor.process_all()

model_version = active_model_version()
processor, feedback_mgr = init_system(model_version)
tickets = get_processed_tickets(processor, model_version)

# Chỉnh sửa của nhân viên có hiệu lực ngay, không cần chờ train lại
corrections = feedback_mgr.latest_by_ticket()
for t in tickets:
    c = corrections.get(str(t['id']))
    if c and c.get('corrected_department'):
        t['category'] = c['corrected_department']
        t['human_reviewed'] = True
df_tickets = pd.DataFrame(tickets) # Chuyển thành DataFrame để vẽ biểu đồ dễ hơn

# --- TẠO 2 TABS CHUYÊN NGHIỆP ---
tab_management, tab_analytics, tab_admin = st.tabs(["Triage & Management", "Analytics Hub", "Admin Dashboard"])

# Xóa @st.fragment ở đây để tránh lỗi lồng fragment (nested fragment)
def ticket_actions_fragment(ticket, processor, feedback_mgr, selected_id):
    # --- New Feature: GenAI Auto-Reply ---
    st.subheader("Auto-Reply Draft")

    draft_key = f"draft_content_{selected_id}"
    if draft_key not in st.session_state:
        st.session_state[draft_key] = ""

    edit_key = f"editable_draft_{selected_id}"
    if st.button("Generate AI Reply", key=f"generate_draft_{selected_id}"):
        st.session_state[draft_key] = ""  # Clear previous draft
        st.session_state.pop(edit_key, None)  # otherwise the text area keeps the old edited draft
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
                st.session_state[draft_key] = full_response

        except Exception as e:
            st.error(f"Failed to generate AI response. Details: {e}")
            st.session_state[draft_key] = "" # Clear on error

    # Display the editable text area if a draft exists
    if st.session_state[draft_key]:
        edited_reply = st.text_area("Review and edit before sending:", value=st.session_state[draft_key], height=250, key=edit_key)
        # ponytail: chưa nối SMTP/helpdesk; chỉ ghi lại bản phản hồi đã duyệt. Nối gửi email thật tại đây.
        if st.button("Mark as Sent", key=f"send_draft_{selected_id}"):
            feedback_mgr.log_reply(selected_id, edited_reply)
            st.success("Reply logged to data/sent_replies.json (email sending is not connected yet).")
            st.session_state[draft_key] = ""
            st.session_state.pop(edit_key, None)

    # --- KHU VỰC HUMAN-IN-THE-LOOP ---
    st.subheader("Human Intervention Panel")
    with st.container():
        col_a, col_b = st.columns(2)
        with col_a:
            departments = list(processor.policy.get("department_rules", {}))
            current = ticket.get('category')
            new_dept = st.selectbox(
                "Correct Department (If AI is wrong):",
                departments,
                index=departments.index(current) if current in departments else None,
                key=f"new_dept_{selected_id}",
            )
            reason = st.text_area("Reason for Change:", key=f"reason_{selected_id}")
        with col_b:
            if st.button("Save Correction & Route", key=f"save_correction_{selected_id}"):
                if not new_dept:
                    st.error("Please select the correct department.")
                else:
                    feedback_mgr.save_correction(selected_id, ticket.get('category', ''), new_dept, reason, ticket['email_content'])
                    st.success("Correction saved. It applies now and will be learned on the next retrain (Admin Dashboard).")

# ==========================================
# TAB 1: TICKET MANAGEMENT 
# ==========================================

def render_ticket_management():
    st.sidebar.header("Ticket Explorer")
    ticket_ids = df_tickets['id'].tolist()
    
    # 1. Khởi tạo session state trực tiếp cho Selectbox
    if "ticket_selector" not in st.session_state:
        st.session_state.ticket_selector = ticket_ids[0] if ticket_ids else None

    # 2. NÚT ĐIỀU HƯỚNG: Thay đổi trực tiếp giá trị của ticket_selector
    current_id = st.session_state.ticket_selector
    current_index = ticket_ids.index(current_id) if current_id in ticket_ids else 0

    col_prev, col_next = st.sidebar.columns(2)
    with col_prev:
        if st.button("Trước", use_container_width=True):
            st.session_state.ticket_selector = ticket_ids[max(0, current_index - 1)]
    with col_next:
        if st.button("Sau", use_container_width=True):
            st.session_state.ticket_selector = ticket_ids[min(len(ticket_ids) - 1, current_index + 1)]

    # 3. Hiển thị Selectbox cực kỳ Clean (Streamlit sẽ tự động đồng bộ)
    selected_id = st.sidebar.selectbox("Select a Ticket to Inspect:", ticket_ids, key="ticket_selector")

    # Bọc toàn bộ nội dung chi tiết ticket trong một container có thể cuộn.
    # Cách này sẽ giữ cho trang chính không bị cuộn lên đầu mỗi khi bạn chọn ticket khác.
    # Bạn có thể điều chỉnh giá trị `height` để phù hợp với màn hình của mình.
    with st.container(height=800):
        # Lấy ticket đang chọn
        if selected_id:
            ticket = next((t for t in tickets if t['id'] == selected_id), None)
            if ticket:
                confidence = float(ticket.get('confidence_score', 0))
        
                st.subheader(f"Ticket Detail: {ticket['id']}")
                # Warning if AI is not confident
                if ticket.get('human_reviewed'):
                    st.success(f"Human reviewed: routed to {ticket['category']}.")
                elif confidence < CONFIDENCE_THRESHOLD:
                    st.error(f"Human Review Required: AI Confidence Score is below threshold ({CONFIDENCE_THRESHOLD:.0%}).")
        
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Customer:** {ticket.get('customer', 'N/A')}")
                    st.write(f"**Created at:** {ticket.get('timestamp', 'N/A')}")
                    st.write(f"**Content:** {ticket['email_content']}")
                with col2:
                    st.warning(f"**Status:** {ticket.get('status', 'N/A')}")
                    st.write(f"**Category:** {ticket.get('category', 'N/A')}")
                    st.metric("Confidence Score", f"{confidence*100:.1f}%")
                    
                    # Show all probabilities in a clean dropdown
                    if "category_probabilities" in ticket and ticket["category_probabilities"]:
                        with st.expander("View All Category Probabilities"):
                            sorted_probs = sorted(ticket["category_probabilities"].items(), key=lambda x: x[1], reverse=True)
                            for cat, p in sorted_probs:
                                st.write(f"- **{cat}**: {p*100:.1f}%")
        
                if ticket.get('routed_by_correction'):
                    match = ticket['routed_by_correction']
                    st.info(f"Routed like human-corrected ticket {match['ticket_id']} ({match['similarity']:.0%} similar).")

                if st.button("Show similar past tickets", key=f"similar_{selected_id}"):
                    try:
                        from modules.vector_store import find_similar
                        with st.spinner("Searching similar tickets (first search loads the index)..."):
                            hits = find_similar([ticket['email_content']], n=6, exclude_ids=[selected_id])[0]
                        # Bỏ các vé trùng y hệt nội dung (dữ liệu cũ có vé lặp)
                        hits = [h for h in hits if h['text'] != ticket['email_content']][:5]
                        for h in hits:
                            source = "🧑 Human correction" if h['source'] == 'human_correction' else "Dataset"
                            st.markdown(f"**{h['similarity']:.0%} similar** · {h['category']} · {source}  \n{h['text']}")
                            if h.get('response'):
                                with st.expander("Agent reply"):
                                    st.write(h['response'])
                    except Exception as e:
                        st.error(f"Similar ticket search failed: {e}")

                st.divider()
                # Gọi Fragment xử lý các hành động (Generate AI Reply, Save Correction)
                ticket_actions_fragment(ticket, processor, feedback_mgr, selected_id)

# Hiển thị Fragment vào trong tab_management
with tab_management:
    render_ticket_management()

# ==========================================
# TAB 2: ANALYTICS HUB (Level 2: Data Visualization)
# ==========================================
with tab_analytics:
    st.subheader("System Performance & Traffic Analysis")
    
    # Tính KPI
    kpi = feedback_mgr.calculate_kpi(total_processed=len(tickets), ticket_ids=[t['id'] for t in tickets])
    
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

# ==========================================
# TAB 3: ADMIN DASHBOARD (Model Versioning)
# ==========================================
with tab_admin:
    st.subheader("Model Training & Version Control")
    st.write("Tải lên dữ liệu mới để huấn luyện mô hình. Hệ thống sẽ tự động lưu phiên bản.")

    uploaded_file = st.file_uploader("Upload dataset mới (CSV, JSON, PDF, DOCX)", type=["csv", "json", "pdf", "docx"])

    def start_background_training(path):
        if is_training():
            st.warning("Một lần train khác đang chạy. Vui lòng đợi nó hoàn tất (xem Terminal).")
            return
        # Cho phép thread sử dụng context của Streamlit
        from streamlit.runtime.scriptrunner import add_script_run_ctx
        st.info("Đang huấn luyện mô hình ngầm (Background). Bạn có thể tiếp tục sử dụng hệ thống! Hãy xem Terminal để biết tiến độ.")
        thread = threading.Thread(target=train_new_model_background, args=(path,))
        add_script_run_ctx(thread)
        thread.start()

    col_train, col_retrain = st.columns(2)
    with col_train:
        if st.button("Train New Model", type="primary"):
            if uploaded_file:
                # Lưu file được upload vào folder data
                temp_path = f"data/uploaded_train_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                start_background_training(temp_path)
            else:
                st.error("Vui lòng chọn file CSV trước khi Train!")
    with col_retrain:
        if st.button(f"Retrain with {len(corrections)} corrections"):
            start_background_training("data/tickets.csv")
            
    st.divider()
    st.subheader("Model Versions (Lịch sử huấn luyện)")
    try:
        with open("models/metadata.json", "r") as f:
            meta = json.load(f)
            
        st.success(f"**Phiên bản đang hoạt động (Active):** {meta.get('active_version', 'N/A')}")
        
        versions = meta.get("versions", {})
        if versions:
            v_df = [{"Version": k, "Held-out Accuracy": f"{v['accuracy']*100:.2f}%" if v.get('accuracy') is not None else "N/A", "Training Rows": v.get("training_rows", 0), "Created At": v.get("created_at", "")[:19].replace("T", " ")} for k, v in versions.items()]
            st.dataframe(pd.DataFrame(v_df).sort_values("Created At", ascending=False), use_container_width=True)
    except FileNotFoundError:
        st.info("Chưa có phiên bản model nào được lưu trữ.")