import pandas as pd
from modules.llm_client import chat_completion

def auto_label_csv(file_path="data/tickets.csv", max_samples=10):
    print(f"🚀 Bắt đầu Auto-Labeling bằng LLM (Giới hạn {max_samples} vé để tiết kiệm)...")
    df = pd.read_csv(file_path)
    categories = ["Logistics", "Billing", "Technical", "Sales", "General Inquiry"]
    
    labeled_count = 0
    for index, row in df.iterrows():
        if labeled_count >= max_samples:
            print(f"\n🛑 Đã đạt giới hạn {max_samples} vé. Dừng gọi LLM để tiết kiệm chi phí!")
            break
            
        # Nếu ô category bị rỗng (NaN)
        if pd.isna(row.get('category')) or str(row.get('category')).strip() == "":
            print(f"Đang nhờ LLM phân loại ticket: {row['id']}...")
            prompt = f"Phân loại nội dung sau vào đúng MỘT trong 5 danh mục: {', '.join(categories)}.\nNội dung: '{row['email_content']}'\n\nYÊU CẦU: CHỈ xuất ra ĐÚNG 1 TỪ/CỤM TỪ là tên danh mục, TUYỆT ĐỐI KHÔNG giải thích gì thêm."
            
            try:
                # Nhiệt độ thấp (0.1) giúp mô hình trả lời nghiêm túc, không bịa từ
                label = chat_completion(prompt, temperature=0.1).strip()
                df.at[index, 'category'] = label
                print(f"  -> Đã gán nhãn: {label}")
                labeled_count += 1
            except Exception as e:
                print(f"  -> Lỗi: {e}")
                
    # Ghi lại file CSV với các nhãn đã được điền đầy đủ
    df.to_csv(file_path, index=False)
    print("\n✅ Hoàn tất! File CSV đã có đầy đủ dữ liệu. Bạn hãy chạy app, Logistic Regression đã có thể tự train.")

if __name__ == "__main__":
    auto_label_csv()