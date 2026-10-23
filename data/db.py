import sqlite3

def init_db():
    conn = sqlite3.connect('support_system.db')
    cursor = conn.cursor()
    
    # 1. Bảng Categories (Cấu trúc cây)
    cursor.execute('''CREATE TABLE IF NOT EXISTS Categories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        parent_id INTEGER,
                        FOREIGN KEY(parent_id) REFERENCES Categories(id))''')
    
    # 2. Bảng Tickets (Lưu trữ dữ liệu)
    cursor.execute('''CREATE TABLE IF NOT EXISTS Tickets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        description TEXT,
                        category_id INTEGER,
                        priority TEXT,
                        status TEXT DEFAULT 'Open',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(category_id) REFERENCES Categories(id))''')
    
    # Insert cấu trúc cây vào đây luôn
    # (Mình sẽ hướng dẫn bạn cách Insert dữ liệu vào bảng này ở bước sau)
    
    conn.commit()
    conn.close()
    print("Database đã sẵn sàng!")

if __name__ == "__main__":
    init_db()