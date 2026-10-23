import pandas as pd

df = pd.read_csv("data/tickets.csv")

# 1. Thêm chữ 'T' vào trước ID nếu nó đang là số (ví dụ: '3000' -> 'T3000')
df['id'] = df['id'].astype(str).apply(lambda x: f"T{x}" if x.isdigit() else x)

# 2. Sửa lỗi chính tả (viết thường/viết hoa) cho đồng bộ Category
df['category'] = df['category'].replace({'Account management': 'Account Management'})

df.to_csv("data/tickets.csv", index=False)
print("✅ Đã dọn dẹp xong 100 dòng cũ! Bạn hãy load lại giao diện nhé.")