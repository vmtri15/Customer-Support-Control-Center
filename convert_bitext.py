"""
Chuyển Bitext Customer Support dataset sang định dạng data/tickets.csv.
Nguồn: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
Run: python convert_bitext.py
"""
import numpy as np
import pandas as pd

SRC = "data/bitext/bitext_27k.csv"
OUT = "data/tickets.csv"
PER_DEPT = 500  # giới hạn mỗi phòng ban: cân bằng lớp và giữ UI/train nhanh

INTENT_TO_DEPT = {
    # ACCOUNT
    "create_account": "Account Management",
    "delete_account": "Account Management",
    "edit_account": "Account Management",
    "recover_password": "Account Management",
    "switch_account": "Account Management",
    # Bitext không có intent kỹ thuật; lỗi đăng ký là gần nhất
    "registration_problems": "Technical",
    # CANCEL / INVOICE / PAYMENT
    "check_cancellation_fee": "Billing",
    "check_invoice": "Billing",
    "get_invoice": "Billing",
    "check_payment_methods": "Billing",
    "payment_issue": "Billing",
    # DELIVERY / SHIPPING / ORDER
    "delivery_options": "Logistics",
    "delivery_period": "Logistics",
    "change_shipping_address": "Logistics",
    "set_up_shipping_address": "Logistics",
    "track_order": "Logistics",
    "change_order": "Logistics",
    "cancel_order": "Logistics",
    "place_order": "Sales",
    # REFUND
    "check_refund_policy": "Returns",
    "get_refund": "Returns",
    "track_refund": "Returns",
    # CONTACT / FEEDBACK / SUBSCRIPTION
    "contact_customer_service": "General Inquiry",
    "contact_human_agent": "General Inquiry",
    "complaint": "General Inquiry",
    "review": "General Inquiry",
    "newsletter_subscription": "General Inquiry",
}


def main():
    df = pd.read_csv(SRC)
    df["category"] = df["intent"].map(INTENT_TO_DEPT)
    unmapped = df.loc[df["category"].isna(), "intent"].unique()
    assert len(unmapped) == 0, f"Unmapped intents: {unmapped}"

    df = df.sample(frac=1, random_state=42).groupby("category").head(PER_DEPT)

    rng = np.random.default_rng(42)
    n = len(df)
    seconds_ago = rng.integers(0, 90 * 24 * 3600, n)
    timestamps = pd.Timestamp("2026-10-23") - pd.to_timedelta(seconds_ago, unit="s")
    bitext = pd.DataFrame({
        "id": [f"B{i:05d}" for i in range(n)],
        "customer_name": [f"Customer {i}" for i in range(n)],
        "email_content": df["instruction"].values,
        "tier": rng.choice(["Bronze", "Silver", "Gold"], n),
        "intent": df["intent"].values,
        "confidence_score": None,
        "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "status": rng.choice(["Open", "Pending"], n),
        "category": df["category"].values,
    })

    # Giữ các vé cũ đã có nhãn (tiếng Việt + ví dụ Technical mà Bitext thiếu); bỏ vé B* để chạy lại không bị trùng
    old = pd.read_csv(OUT)
    old = old[old["category"].notna() & ~old["id"].astype(str).str.startswith("B")]

    out = pd.concat([old, bitext], ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"✅ Wrote {len(out)} tickets ({len(old)} existing labelled + {n} Bitext) to {OUT}")
    print(out["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
