"""
Chuyển Bitext Customer Support dataset sang định dạng data/tickets.csv,
kèm vé Technical đa dạng (EN + VN) mà Bitext không có.
Nguồn: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
Run: python convert_bitext.py
"""
import itertools

import numpy as np
import pandas as pd

SRC = "data/bitext/bitext_27k.csv"
OUT = "data/tickets.csv"
PER_DEPT = 500  # giới hạn mỗi phòng ban: cân bằng lớp và giữ UI/train nhanh
# Technical của Bitext chỉ là lỗi đăng ký: lấy ít hơn, phần còn lại là vé tự sinh
DEPT_CAPS = {"Technical": 250}
TECH_PER_LANG = 125

TECH_PARTS = {
    "en": (
        ["The app", "Your website", "The mobile app on my iPhone", "The Android app", "The cart page",
         "The search page", "Your site", "The product page", "The app on my tablet", "The web dashboard"],
        ["keeps crashing", "won't load at all", "is extremely slow", "shows a 500 error", "freezes",
         "shows a blank white screen", "times out", "throws an error", "keeps logging me out",
         "shows a 404 page not found", "won't open", "gets stuck on the loading screen",
         "displays broken images", "is not responding"],
        ["when I open it", "after the latest update", "every time I tap the cart", "since this morning",
         "when I try to upload a photo", "whenever I search for a product", "on Chrome and Safari",
         "after I click the submit button", "on my phone", "for the past two days", ""],
        ["Please fix this.", "Is the server down?", "This is really frustrating.", "Can someone look into it?",
         "I already tried clearing my cache.", "Reinstalling didn't help.", "Is this a known bug?", ""],
    ),
    "vn": (
        ["Ứng dụng", "Trang web", "App trên iPhone của mình", "App Android", "Trang giỏ hàng",
         "Trang tìm kiếm", "Website bên bạn", "Trang sản phẩm", "App trên máy tính bảng", "Trang quản lý"],
        ["cứ bị văng ra", "không tải được", "chạy chậm kinh khủng", "báo lỗi 500", "bị đơ",
         "hiện màn hình trắng", "bị treo", "báo lỗi kết nối máy chủ", "tự đăng xuất liên tục",
         "báo lỗi 404", "không mở lên được", "cứ quay vòng tải mãi không xong",
         "không hiển thị hình ảnh", "không phản hồi"],
        ["khi mình mở lên", "sau bản cập nhật mới nhất", "mỗi lần bấm vào giỏ hàng", "từ sáng tới giờ",
         "khi mình tải ảnh lên", "mỗi khi tìm kiếm sản phẩm", "trên cả Chrome lẫn Safari",
         "sau khi bấm nút gửi", "trên điện thoại của mình", "mấy hôm nay rồi", ""],
        ["Sửa giúp mình với.", "Server bị sập à?", "Bực mình quá.", "Bên kỹ thuật kiểm tra giúp mình nhé.",
         "Mình đã xóa cache rồi mà vẫn bị.", "Cài lại app cũng không được.", "Đây có phải lỗi chung không?", ""],
    ),
}


def technical_texts(rng) -> list[str]:
    """Ghép thiết bị/trang + triệu chứng + ngữ cảnh + câu kết; lấy mẫu không trùng lặp."""
    texts = []
    for surfaces, symptoms, triggers, closings in TECH_PARTS.values():
        combos = list(itertools.product(surfaces, symptoms, triggers, closings))
        for i in rng.choice(len(combos), TECH_PER_LANG, replace=False):
            surface, symptom, trigger, closing = combos[i]
            sentence = " ".join(p for p in (surface, symptom, trigger) if p) + "."
            texts.append(f"{sentence} {closing}".strip())
    return texts

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

    df = df.sample(frac=1, random_state=42)
    caps = df["category"].map(lambda c: DEPT_CAPS.get(c, PER_DEPT))
    df = df[df.groupby("category").cumcount() < caps]

    rng = np.random.default_rng(42)
    tech = technical_texts(rng)
    texts = list(df["instruction"]) + tech
    intents = list(df["intent"]) + ["technical_issue"] * len(tech)
    categories = list(df["category"]) + ["Technical"] * len(tech)
    n = len(texts)
    seconds_ago = rng.integers(0, 90 * 24 * 3600, n)
    timestamps = pd.Timestamp("2026-10-23") - pd.to_timedelta(seconds_ago, unit="s")
    bitext = pd.DataFrame({
        "id": [f"B{i:05d}" for i in range(n)],
        "customer_name": [f"Customer {i}" for i in range(n)],
        "email_content": texts,
        "tier": rng.choice(["Bronze", "Silver", "Gold"], n),
        "intent": intents,
        "confidence_score": None,
        "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "status": rng.choice(["Open", "Pending"], n),
        "category": categories,
    })

    # Giữ các vé cũ đã có nhãn (tiếng Việt + ví dụ Technical mà Bitext thiếu); bỏ vé B* để chạy lại không bị trùng
    old = pd.read_csv(OUT)
    old = old[old["category"].notna() & ~old["id"].astype(str).str.startswith("B")]

    out = pd.concat([old, bitext], ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"✅ Wrote {len(out)} tickets ({len(old)} existing labelled + {n - len(tech)} Bitext + {len(tech)} generated Technical) to {OUT}")
    print(out["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
