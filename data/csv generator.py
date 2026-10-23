import pandas as pd
import random
from datetime import datetime, timedelta

def generate_more_tickets(n=500, start_id_num=2500):
    # Template mới, tập trung vào sự đa dạng ngôn ngữ và sắc thái
    content_pool = [
    # --- Logistics ---
    ("My order #{id} is literally 5 days late. Where is my stuff?", "Logistics", "logistic"),
    ("Đơn hàng #{id} của mình bị trễ quá lâu rồi, giờ không biết đang ở đâu luôn.", "Logistics", "logistic"),
    ("Tracking shows delivered but nothing at my door. Can you check this?", "Logistics", "logistic"),
    ("Giao hàng chậm quá, mình cần gấp mà chưa thấy gì. Check giúp mình với.", "Logistics", "logistic"),
    ("Is it possible to expedite my order #{id}? I need it for a gift.", "Logistics", "logistic"),
    ("Hàng đã tới kho bưu điện rồi mà sao shipper chưa giao cho mình nữa?", "Logistics", "logistic"),
    ("My tracking link is broken, it just shows a blank white page.", "Logistics", "logistic"),
    ("Shipper làm móp hộp hàng của mình, mình muốn phản ánh vấn đề này.", "Logistics", "logistic"),
    ("Why is my package moving between hubs for 3 days now?", "Logistics", "logistic"),
    ("Sắp hết thời gian dự kiến giao hàng rồi mà vẫn chưa thấy cập nhật gì cả.", "Logistics", "logistic"),

    # --- Billing ---
    ("I was overcharged on my last invoice. Please fix this.", "Billing", "billing"),
    ("Tại sao đơn hàng #{id} lại bị trừ tiền 2 lần vậy? Hoàn tiền cho tôi ngay.", "Billing", "billing"),
    ("Still waiting for my refund. It's been 2 weeks already. Ridiculous.", "Billing", "billing"),
    ("Mình chưa nhận được hóa đơn VAT cho đơn hàng tháng trước, gửi lại giúp nhé.", "Billing", "billing"),
    ("The discount code did not apply to my final checkout price.", "Billing", "billing"),
    ("Tôi muốn thay đổi phương thức thanh toán cho các đơn hàng sau này.", "Billing", "billing"),
    ("How do I update my credit card info in the system?", "Billing", "billing"),
    ("Tại sao thẻ của tôi bị từ chối trong khi vẫn còn hạn mức?", "Billing", "billing"),
    ("I paid for premium but I'm still seeing ads. Please help.", "Billing", "billing"),
    ("Mình thanh toán qua ví điện tử bị lỗi, tiền đã trừ nhưng đơn vẫn chưa xác nhận.", "Billing", "billing"),

   # --- Technical (Lỗi hệ thống) ---
    ("App keeps crashing on startup. Very frustrating.", "Technical", "technical"),
    ("Website báo lỗi 500, không truy cập được.", "Technical", "technical"),
    ("The website is loading incredibly slowly today, is the server down?", "Technical", "technical"),
    ("I'm getting 'Unauthorized' error whenever I click on my orders.", "Technical", "technical"),
    ("Cứ bấm vào giỏ hàng là ứng dụng bị văng (force close).", "Technical", "technical"),

    # --- Account Management (Hỗ trợ tài khoản) ---
    ("I've been trying to reset my password for an hour, no email received.", "Account Management", "account management"),
    ("Tài khoản bị khóa vô lý quá, tôi cần hỗ trợ mở lại gấp.", "Account Management", "account management"),
    ("Mình quên mất username rồi, khôi phục bằng cách nào?", "Account Management", "account management"),
    ("My account is showing the wrong name, how can I edit my profile?", "Account Management", "account management"),
    ("Tại sao mình không nhận được thông báo đẩy của app?", "Account Management", "account management"),

    # --- Sales / General ---
    ("Do you offer bulk discounts for enterprise clients?", "Sales", "sales"),
    ("Mình muốn mua sỉ, bạn có bảng giá hay chiết khấu không?", "Sales", "sales"),
    ("Are you guys open to collaborations?", "General Inquiry", "general inquiry"),
    ("Làm sao để tham gia chương trình khách hàng thân thiết vậy?", "General Inquiry", "general inquiry"),
    ("I'm interested in becoming a reseller of your products.", "Sales", "sales"),
    ("Bên mình muốn hợp tác làm event cùng thương hiệu của bạn.", "General Inquiry", "general inquiry"),
    ("What are the store hours for your physical locations?", "General Inquiry", "general inquiry"),
    ("Có bảng size chart (bảng hướng dẫn chọn size) cho sản phẩm này không?", "General Inquiry", "general inquiry"),
    ("I'm thinking of upgrading to Gold tier, what are the extra perks?", "Sales", "sales"),
    ("Mình cần thông tin liên hệ của bộ phận đối tác để hợp tác dài hạn.", "General Inquiry", "general inquiry"),
    
    # --- Returns ---
    ("I need to return this item, it doesn't fit my size.", "Returns", "returns"),
    ("Sản phẩm bị lỗi kỹ thuật, mình muốn đổi cái mới.", "Returns", "returns"),
    ("The package arrived damaged, I want a replacement.", "Returns", "returns"),
    ("Hàng bị lỗi, mình muốn hoàn trả và nhận lại tiền.", "Returns", "returns"),
    ("Can I exchange this for a different color?", "Returns", "returns")
]
    
    new_tickets = []
    names = [
    # English
    "James Anderson", "Olivia Smith", "William Taylor", "Emma Johnson", "Benjamin Brown",
    "Sophia Miller", "Henry Wilson", "Charlotte Davis", "Lucas Martinez", "Amelia Garcia",
    "Alexander Rodriguez", "Evelyn Martinez", "Daniel Hernandez", "Harper Lopez", "Matthew Gonzalez",
    "Elizabeth Wilson", "Joseph Anderson", "Grace Thomas", "Samuel Taylor", "Victoria Moore",
    
    # Vietnamese
    "Nguyen Van An", "Tran Thi Mai", "Le Minh Tuan", "Pham Thanh Thao", "Hoang Duc Anh",
    "Vu Thi Lan", "Dang Minh Tri", "Bui Thu Ha", "Do Xuan Bach", "Nguyen Thi Bich",
    "Phan Quoc Bao", "Vo Minh Phuong", "Nguyen Hoang Nam", "Tran Minh Chau", "Le Thu Huong",
    "Pham Duc Minh", "Hoang Thi Yen", "Vu Minh Khoi", "Dang Thu Trang", "Bui Van Long",
    
    # European
    "Lukas Schmidt", "Elena Rossi", "Mateo Garcia", "Freja Nielsen", "Sven Johansson",
    "Isabella Müller", "Hugo Dubois", "Sofia Silva", "Jakub Kowalski", "Anna Novak",
    "Maximilian Weber", "Clara Dubois", "Erik Hansen", "Martina Costa", "Jonas Lindberg",
    "Alice Lefebvre", "Alessandro Conti", "Katarina Petrov", "Oskar Berg", "Beatriz Santos"
]
    tiers = ["Bronze", "Silver", "Gold"]
    
    for i in range(n):
        tpl = random.choice(content_pool)
        timestamp = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d %H:%M:%S")
        
        content = tpl[0]
        # Chỉ format nếu chuỗi template có chứa placeholder {id}
        if "{id}" in content:
            content = content.format(id=random.randint(1000, 9999))
        
        # LOGIC SỬA LẠI ĐÂY:
        # Nếu i < 200 thì gán tpl[1] (Category), ngược lại gán chuỗi rỗng ""
        category_val = tpl[1] if i < 200 else ""
        
        new_tickets.append({
            "id": f"T{start_id_num + i}",
            "customer_name": random.choice(names),
            "email_content": content,
            "tier": random.choice(tiers),
            "intent": tpl[2],
            "confidence_score": "",
            "timestamp": timestamp,
            "status": random.choice(["Open", "Closed", "Pending"]),
            "category": category_val
        })
        
    return pd.DataFrame(new_tickets)

# Tạo file
df = generate_more_tickets(500, start_id_num=2500)
df.to_csv("data/semi_supervised_tickets.csv", index=False)

print("Đã tạo xong: 200 dòng có Category, 300 dòng để trống Category!")