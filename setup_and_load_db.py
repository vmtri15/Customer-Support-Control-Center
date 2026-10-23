import sqlite3
import pandas as pd
import os

DB_NAME = 'support_system.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Xóa bảng cũ nếu có để tạo lại từ đầu
    cursor.execute('DROP TABLE IF EXISTS Tickets')
    cursor.execute('DROP TABLE IF EXISTS Categories')

    # 1. Tạo bảng Categories (Cấu trúc phân cấp)
    cursor.execute('''CREATE TABLE Categories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        parent_id INTEGER,
                        FOREIGN KEY(parent_id) REFERENCES Categories(id))''')
    
    # 2. Tạo bảng Tickets với đầy đủ tất cả các cột từ file CSV
    cursor.execute('''CREATE TABLE Tickets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticket_id TEXT,
                        customer_name TEXT,
                        customer_email TEXT,
                        customer_age INTEGER,
                        customer_gender TEXT,
                        product_purchased TEXT,
                        date_of_purchase TEXT,
                        ticket_type TEXT,
                        category_id INTEGER,
                        ticket_subject TEXT,
                        ticket_description TEXT,
                        ticket_status TEXT,
                        resolution TEXT,
                        ticket_priority TEXT,
                        ticket_channel TEXT,
                        first_response_time TEXT,
                        time_to_resolution TEXT,
                        customer_satisfaction_rating INTEGER,
                        FOREIGN KEY(category_id) REFERENCES Categories(id))''')
    
    # 3. Insert dữ liệu mẫu cho Category (6 nhóm cha)
    categories = [('Billing', None), ('Returns/Refunds', None), ('Technical', None), 
                  ('Sales', None), ('Account Management', None), ('General Inquiry', None)]
    
    cursor.executemany("INSERT OR IGNORE INTO Categories (name, parent_id) VALUES (?, ?)", categories)
    
    conn.commit()
    conn.close()
    print("1. Database và Schema đã được tạo lại thành công!")

def load_csv_to_db(csv_path):
    if not os.path.exists(csv_path):
        print(f"Lỗi: Không tìm thấy file CSV tại '{csv_path}'")
        return

    df = pd.read_csv(csv_path)
    
    # Xử lý các giá trị trống (NaN) thành None để SQLite có thể hiểu là NULL
    df = df.where(pd.notnull(df), None)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Logic ánh xạ (Mapping) từ Ticket Type sang 6 nhóm
    mapping = {
        'Technical issue': 3,
        'Billing inquiry': 1,
        'Refund request': 2,
        'Cancellation request': 2,
        'Product inquiry': 4
    }
    
    for _, row in df.iterrows():
        cat_id = mapping.get(row.get('Ticket Type'), 6)
        
        # Insert đầy đủ các cột tương ứng từ CSV vào SQLite
        cursor.execute('''
            INSERT INTO Tickets (
                ticket_id, customer_name, customer_email, customer_age, customer_gender,
                product_purchased, date_of_purchase, ticket_type, category_id,
                ticket_subject, ticket_description, ticket_status, resolution,
                ticket_priority, ticket_channel, first_response_time, time_to_resolution,
                customer_satisfaction_rating
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row.get('Ticket ID'), row.get('Customer Name'), row.get('Customer Email'), 
            row.get('Customer Age'), row.get('Customer Gender'), row.get('Product Purchased'), 
            row.get('Date of Purchase'), row.get('Ticket Type'), cat_id,
            row.get('Ticket Subject'), row.get('Ticket Description'), row.get('Ticket Status'), 
            row.get('Resolution'), row.get('Ticket Priority'), row.get('Ticket Channel'), 
            row.get('First Response Time'), row.get('Time to Resolution'), row.get('Customer Satisfaction Rating')
        ))
    
    conn.commit()
    conn.close()
    print("2. Đã nạp xong toàn bộ dữ liệu từ CSV vào Database!")

if __name__ == '__main__':
    csv_file = 'data/customer_support_tickets.csv'
    init_db()
    load_csv_to_db(csv_file)
    print("Hoàn tất quá trình Setup Database và Load Data!")
