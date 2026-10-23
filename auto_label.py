import json

import pandas as pd
from modules.processor import llm_label

def auto_label_csv(file_path="data/tickets.csv", max_samples=10, policy_path="data/company_policy.json"):
    print(f"🚀 Bắt đầu Auto-Labeling bằng LLM (Giới hạn {max_samples} vé để tiết kiệm)...")
    df = pd.read_csv(file_path)
    with open(policy_path, encoding="utf-8") as f:
        categories = list(json.load(f)["department_rules"])

    labeled_count = 0
    for index, row in df.iterrows():
        if labeled_count >= max_samples:
            print(f"\n🛑 Đã đạt giới hạn {max_samples} vé. Dừng gọi LLM để tiết kiệm chi phí!")
            break

        # Nếu ô category bị rỗng (NaN)
        if pd.isna(row.get('category')) or str(row.get('category')).strip() == "":
            print(f"Đang nhờ LLM phân loại ticket: {row['id']}...")
            try:
                label = llm_label(row['email_content'], categories)
                labeled_count += 1
                if label is None:
                    print("  -> LLM trả lời không hợp lệ, bỏ qua")
                    continue
                df.at[index, 'category'] = label
                print(f"  -> Đã gán nhãn: {label}")
            except Exception as e:
                print(f"  -> Lỗi: {e}")

    # Ghi lại file CSV với các nhãn đã được điền
    df.to_csv(file_path, index=False)
    print("\n✅ Hoàn tất! Hãy train lại mô hình SVM trong Admin Dashboard.")

if __name__ == "__main__":
    auto_label_csv()
