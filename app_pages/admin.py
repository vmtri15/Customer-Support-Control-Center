import json
import threading

import pandas as pd
import streamlit as st

from app_pages import common
from modules.processor import is_training, train_new_model_background

common.current_agent()
corrections = common.get_feedback().latest_by_ticket()

st.title("Admin")
st.subheader("Model Training & Version Control")
st.write("Tải lên dữ liệu mới để huấn luyện mô hình. Hệ thống sẽ tự động lưu phiên bản. "
         "Phiên bản mới được dùng cho các vé mới.")

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
