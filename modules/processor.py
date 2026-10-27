import json
import re
import os
import shutil
import datetime
from pathlib import Path
import threading

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

CONFIDENCE_THRESHOLD = 0.65
CORRECTION_MATCH_SIMILARITY = 0.85  # cosine similarity để vé độ tự tin thấp đi theo một chỉnh sửa của nhân viên

# Khởi tạo underthesea 1 lần duy nhất ở mức Global để tối ưu tốc độ
try:
    from underthesea import word_tokenize as vn_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False
    print("⚠️ Cảnh báo: Vui lòng cài đặt underthesea (pip install underthesea)")

_tokenize_lock = threading.Lock()


def llm_label(text: str, categories: list[str]) -> str | None:
    """Xin nhãn từ LLM; trả về None nếu câu trả lời không chứa đúng 1 danh mục hợp lệ."""
    from modules.llm_client import chat_completion
    prompt = f"Phân loại nội dung sau vào đúng MỘT trong {len(categories)} danh mục: {', '.join(categories)}.\nNội dung: '{text}'\n\nYÊU CẦU: CHỈ xuất ra ĐÚNG 1 TỪ/CỤM TỪ là tên danh mục, TUYỆT ĐỐI KHÔNG giải thích gì thêm."
    answer = chat_completion(prompt, temperature=0.1).lower()
    # Chấp nhận "Billing." hay "Category: Billing", từ chối câu trả lời mơ hồ
    matches = [c for c in categories if c.lower() in answer]
    return matches[0] if len(matches) == 1 else None

class MLClassifier:
    """A ML model using TF-IDF and Support Vector Machine (SVM) with Active Learning."""
    def __init__(self, training_data_path: str = "data/tickets.csv", force_retrain: bool = False):
        import joblib
        
        self.models_dir = Path("models")
        self.metadata_path = self.models_dir / "metadata.json"
        self.archive_dir = Path("data/archive")
        self.corrections_path = Path("data/corrections.json")

        self.models_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = "data/svm_model.joblib" # fallback
        self.vectorizer_path = "data/tfidf_vectorizer.joblib" # fallback
        self.active_version = None
        
        self._load_metadata()
        
        if not force_retrain and os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.is_trained = True
            print(f"✅ Loaded ML classifier (Version: {self.active_version or 'Default'})")
        else:
            # Bỏ stop_words='english' vì dữ liệu có tiếng Việt, giữ ngram để bắt cụm từ
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            # Sử dụng LinearSVC kết hợp CalibratedClassifierCV để tính xác suất từ khoảng cách siêu phẳng
            base_svm = LinearSVC(class_weight='balanced', max_iter=2000)
            self.model = CalibratedClassifierCV(estimator=base_svm, cv=5, method='sigmoid')
            self.is_trained = False
            self._train(training_data_path)

    def _load_metadata(self):
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    meta = json.load(f)
                active_v = meta.get("active_version")
                if active_v and active_v in meta.get("versions", {}):
                    v_info = meta["versions"][active_v]
                    if os.path.exists(v_info["model_path"]) and os.path.exists(v_info["vectorizer_path"]):
                        self.model_path = v_info["model_path"]
                        self.vectorizer_path = v_info["vectorizer_path"]
                        self.active_version = active_v
            except Exception as e:
                print(f"⚠️ Error loading metadata: {e}")

    def _preprocess_text(self, text: str) -> str:
        """Làm sạch văn bản trước khi đưa vào mô hình."""
        if not isinstance(text, str):
            return ""
        text = text.lower() # Chuyển về chữ thường
        # Dùng [^\w\s] để giữ lại các ký tự có dấu của tiếng Việt, xóa dấu câu/ký tự đặc biệt
        text = re.sub(r'[^\w\s]', ' ', text)
        text = text.replace('_', ' ') # Loại bỏ thêm dấu gạch dưới nếu có
        text = re.sub(r'\s+', ' ', text).strip() # Xóa khoảng trắng thừa
        
        # Tách từ tiếng Việt (Word Segmentation) bằng underthesea
        if HAS_UNDERTHESEA:
            with _tokenize_lock:
                text = vn_tokenize(text, format="text")
        return text

    def _read_pdf_text(self, file_path: str) -> str:
        """Trích xuất toàn bộ nội dung text từ file PDF."""
        import pypdf
        text = ""
        try:
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc file PDF {file_path}: {e}")
        return text

    def _read_docx_text(self, file_path: str) -> str:
        """Trích xuất toàn bộ nội dung text từ file DOCX."""
        import docx
        text = ""
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc file DOCX {file_path}: {e}")
        return text

    def _load_dataframe_from_file(self, file_path: str) -> pd.DataFrame | None:
        """Đọc file dữ liệu từ nhiều định dạng khác nhau và chuyển thành DataFrame."""
        suffix = Path(file_path).suffix.lower()
        try:
            if suffix == ".csv":
                return pd.read_csv(file_path)
            elif suffix == ".json":
                # Giả định file JSON có cấu trúc là một danh sách các object, mỗi object là một ticket
                return pd.read_json(file_path, orient="records")
            elif suffix == ".pdf":
                content = self._read_pdf_text(file_path)
                return pd.DataFrame([{"email_content": content, "category": None, "id": Path(file_path).stem}])
            elif suffix == ".docx":
                content = self._read_docx_text(file_path)
                return pd.DataFrame([{"email_content": content, "category": None, "id": Path(file_path).stem}])
            else:
                print(f"⚠️ Định dạng file không được hỗ trợ: {suffix}")
                return None
        except Exception as e:
            print(f"❌ Lỗi nghiêm trọng khi xử lý file {file_path}: {e}")
            return None

    def _apply_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ghi đè nhãn bằng chỉnh sửa của nhân viên (Human-in-the-loop); thêm vé chưa có trong file."""
        from modules.feedback_manager import FeedbackManager
        latest = FeedbackManager(self.corrections_path).latest_by_ticket()
        ids = df['id'].astype(str) if 'id' in df.columns else pd.Series("", index=df.index)

        applied, new_rows = 0, []
        for ticket_id, c in latest.items():
            dept = (c.get("corrected_department") or "").strip()
            if not dept:
                continue
            mask = ids == ticket_id
            if mask.any():
                df.loc[mask, 'category'] = dept
                applied += 1
            elif c.get("email_content"):
                new_rows.append({"id": ticket_id, "email_content": c["email_content"], "category": dept})

        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        if applied or new_rows:
            print(f"🧑‍🏫 Áp dụng {applied} chỉnh sửa, thêm {len(new_rows)} vé từ corrections.json")
        return df

    def _build_model(self, y: pd.Series) -> CalibratedClassifierCV:
        """Calibrated SVM; số fold tự giảm khi có lớp ít mẫu (cv=5 cần ≥5 vé mỗi lớp)."""
        cv = max(2, min(5, int(y.value_counts().min())))
        base_svm = LinearSVC(class_weight='balanced', max_iter=2000)
        return CalibratedClassifierCV(estimator=base_svm, cv=cv, method='sigmoid')

    def _drop_rare_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bỏ danh mục chỉ có 1 vé: không thể cross-validate để calibrate."""
        counts = df['category'].value_counts()
        rare = counts[counts < 2].index.tolist()
        if rare:
            print(f"⚠️ Bỏ qua danh mục chỉ có 1 vé (cần ≥2): {rare}")
            df = df[~df['category'].isin(rare)]
        return df

    def _holdout_accuracy(self, df_final: pd.DataFrame, labeled_index) -> float | None:
        """
        Accuracy trên 20% vé có nhãn thật (gốc + chỉnh sửa của người), giữ lại không train.
        Không chấm trên nhãn do SVM/LLM tự gán vì mô hình sẽ dễ dàng đoán đúng nhãn của chính nó.
        """
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        df_real = df_final.loc[df_final.index.intersection(labeled_index)]
        try:
            _, test_idx = train_test_split(
                df_real.index, test_size=0.2, stratify=df_real['category'], random_state=42
            )
            df_train = df_final.drop(index=test_idx)
            df_test = df_final.loc[test_idx]
            vec = TfidfVectorizer(ngram_range=(1, 2))
            model = self._build_model(df_train['category'])
            model.fit(vec.fit_transform(df_train['processed_content']), df_train['category'])
            preds = model.predict(vec.transform(df_test['processed_content']))
            acc = float(accuracy_score(df_test['category'], preds))
            print(f"📊 Held-out accuracy: {acc:.2%} trên {len(df_test)} vé có nhãn thật")
            return acc
        except ValueError as e:
            # Quá ít mẫu mỗi lớp để chia stratified / CalibratedClassifierCV(cv=5)
            print(f"⚠️ Không đủ dữ liệu có nhãn để đánh giá held-out: {e}")
            return None

    def _train(self, data_path: str):
        try:
            import joblib
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            version_id = f"v{timestamp}"
            version_dir = self.models_dir / version_id
            version_dir.mkdir(parents=True, exist_ok=True)
            
            archive_suffix = Path(data_path).suffix
            archive_path = self.archive_dir / f"dataset_{timestamp}{archive_suffix}"
            if os.path.exists(data_path):
                shutil.copy(data_path, archive_path)

            df_full = self._load_dataframe_from_file(data_path)
            if df_full is None or df_full.empty:
                print("⚠️ Không thể đọc dữ liệu từ file, dừng quá trình train.")
                return

            if 'email_content' in df_full.columns and 'category' in df_full.columns:
                
                # Áp dụng tiền xử lý
                df_full = self._apply_corrections(df_full.copy())
                df_full['processed_content'] = df_full['email_content'].apply(self._preprocess_text)
                
                # Chia tập dữ liệu có nhãn và không có nhãn
                labeled_mask = df_full['category'].notna() & (df_full['category'].astype(str).str.strip() != "")
                df_labeled = self._drop_rare_classes(df_full[labeled_mask])
                df_unlabeled = df_full[~labeled_mask]

                if df_labeled['category'].nunique() < 2:
                    print("⚠️ LỖI: Cần ít nhất 2 danh mục, mỗi danh mục ≥2 vé có nhãn để làm seed data cho SVM.")
                    self.is_trained = False
                    return

                print(f"🚀 Bước 1: Train SVM ban đầu trên {len(df_labeled)} mẫu có sẵn.")
                X_labeled = self.vectorizer.fit_transform(df_labeled['processed_content'])
                y_labeled = df_labeled['category']

                initial_model = self._build_model(y_labeled)
                initial_model.fit(X_labeled, y_labeled)
                
                if len(df_unlabeled) > 0:
                    print(f"🔍 Bước 2: SVM dự đoán cho {len(df_unlabeled)} mẫu chưa có nhãn...")
                    categories = sorted(df_labeled['category'].unique())

                    X_unlabeled = self.vectorizer.transform(df_unlabeled['processed_content'])
                    probs = initial_model.predict_proba(X_unlabeled)
                    preds = initial_model.predict(X_unlabeled)
                    
                    llm_calls = 0
                    max_llm_calls = 20 # Giới hạn gọi LLM để tránh treo máy
                    
                    for i, idx in enumerate(df_unlabeled.index):
                        confidence = float(max(probs[i]))
                        
                        if confidence >= 0.7:
                            df_full.at[idx, 'category'] = preds[i]
                        else:
                            if llm_calls < max_llm_calls:
                                print(f"    ⏳ Xin nhãn từ LLM cho vé {df_full.at[idx, 'id']} (Độ tự tin: {confidence:.2f})...")
                                try:
                                    label = llm_label(df_full.at[idx, 'email_content'], categories)
                                    llm_calls += 1
                                    if label is None:
                                        print(f"      -> LLM trả lời không hợp lệ, dùng SVM: {preds[i]}")
                                        label = preds[i]
                                    else:
                                        print(f"      -> LLM chọn: {label}")
                                    df_full.at[idx, 'category'] = label
                                except Exception as e:
                                    print(f"      -> Lỗi gọi LLM: {e}")
                                    df_full.at[idx, 'category'] = preds[i] # Fallback to SVM
                            else:
                                df_full.at[idx, 'category'] = preds[i]
                                
                    if llm_calls >= max_llm_calls:
                        print(f"⚠️ Đã đạt giới hạn {max_llm_calls} lần gọi LLM. Các vé còn lại được ép dùng SVM để tiết kiệm thời gian.")
                    # Không ghi đè file gốc: lưu bản đã dán nhãn vào archive
                    labeled_path = self.archive_dir / f"labeled_{timestamp}.csv"
                    df_full.drop(columns=['processed_content']).to_csv(labeled_path, index=False)
                    print(f"✅ Đã dán nhãn xong toàn bộ dữ liệu. Lưu tại: {labeled_path}")

                print(f"🎓 Bước 3: Train lại mô hình SVM 'trùm cuối' trên toàn bộ {len(df_full)} mẫu...")
                df_final = self._drop_rare_classes(df_full.dropna(subset=['category', 'email_content']))
                X_final = self.vectorizer.fit_transform(df_final['processed_content'])
                y_final = df_final['category']

                self.model = self._build_model(y_final)
                self.model.fit(X_final, y_final)
                self.is_trained = True
                
                # Save new version
                new_model_path = str(version_dir / "svm_model.joblib")
                new_vec_path = str(version_dir / "tfidf_vectorizer.joblib")
                joblib.dump(self.model, new_model_path)
                joblib.dump(self.vectorizer, new_vec_path)
                
                acc = self._holdout_accuracy(df_final, df_labeled.index)
                
                meta = {"active_version": version_id, "versions": {}}
                if self.metadata_path.exists():
                    try:
                        with open(self.metadata_path, "r") as f:
                            meta = json.load(f)
                    except (json.JSONDecodeError, OSError) as e:
                        # Giữ lại file hỏng thay vì âm thầm ghi đè lịch sử phiên bản
                        backup = self.metadata_path.with_suffix(f".corrupt_{timestamp}.json")
                        shutil.copy(self.metadata_path, backup)
                        print(f"⚠️ metadata.json bị lỗi ({e}); đã sao lưu sang {backup}")
                if "versions" not in meta: meta["versions"] = {}
                meta["versions"][version_id] = {
                    "created_at": datetime.datetime.now().isoformat(),
                    "model_path": new_model_path,
                    "vectorizer_path": new_vec_path,
                    "archive_data_path": str(archive_path),
                    "training_rows": len(df_final),
                    "accuracy": acc
                }
                meta["active_version"] = version_id
                with open(self.metadata_path, "w") as f:
                    json.dump(meta, f, indent=4)
                    
                self.model_path = new_model_path
                self.vectorizer_path = new_vec_path
                self.active_version = version_id
                print(f"🏆 Quá trình Active Learning hoàn tất! Đã lưu phiên bản: {version_id}")
            else:
                print("⚠️ Missing required columns for training ML Classifier.")
        except Exception as e:
            print(f"⚠️ Error training ML Classifier: {e}")

    def predict(self, text: str) -> tuple[str, float, dict]:
        """Predicts the category, returns confidence, and all probabilities."""
        return self.predict_many([text])[0]

    def predict_many(self, texts: list[str]) -> list[tuple[str, float, dict]]:
        """Dự đoán theo lô: một lần transform + predict_proba cho tất cả vé (nhanh hơn nhiều so với từng vé)."""
        default = ("General Inquiry", 0.0, {})
        if not self.is_trained:
            return [default] * len(texts)

        # Bước 1: Tiền xử lý (Preprocessing) trước khi dự đoán
        processed = [self._preprocess_text(t) if isinstance(t, str) else "" for t in texts]
        valid = [i for i, p in enumerate(processed) if p]
        results = [default] * len(texts)
        if not valid:
            return results

        probs = self.model.predict_proba(self.vectorizer.transform([processed[i] for i in valid]))
        classes = [str(c) for c in self.model.classes_]
        for row, i in enumerate(valid):
            best = int(probs[row].argmax())
            prob_dict = dict(zip(classes, map(float, probs[row])))
            results[i] = (classes[best], float(probs[row][best]), prob_dict)
        return results


class TicketProcessor:
    """Process tickets from a CSV file according to the company policy."""

    def __init__(
        self,
        file_path: str | Path,
        policy_path: str | Path | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.policy_path = Path(policy_path or self.file_path.parent / "company_policy.json")
        self.tickets: list[dict] = []
        self.classifier = MLClassifier()

        with self.policy_path.open(encoding="utf-8") as file:
            self.policy = json.load(file)

    def load_tickets(self) -> list[dict]:
        """Load tickets from a CSV file using pandas and return a list of tickets."""
        df = pd.read_csv(self.file_path)
        if 'customer_name' in df.columns:
            df = df.rename(columns={'customer_name': 'customer'})
        self.tickets = df.to_dict(orient="records")
        return self.tickets

    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment based on sentiment_thresholds in the company policy."""
        text_lower = text.lower()
        sentiment_config = self.policy.get("sentiment_thresholds", {})
        critical_keywords = sentiment_config.get("critical_keywords", [])

        negative_keywords = critical_keywords + sentiment_config.get("negative_keywords", [])
        if any(self._keyword_matches(text_lower, keyword) for keyword in negative_keywords):
            return "Negative"

        positive_keywords = sentiment_config.get("positive_keywords", [])
        if any(self._keyword_matches(text_lower, keyword) for keyword in positive_keywords):
            return "Positive"

        return "Neutral"

    def assign_tags(self, text: str) -> list[str]:
        """Assign tags (department) automatically based on department_rules in the company policy."""
        text_lower = text.lower()
        tags: list[str] = []

        for department, keywords in self.policy.get("department_rules", {}).items():
            if any(self._keyword_matches(text_lower, keyword) for keyword in keywords):
                tags.append(department)

        return tags

    def validate_routing(self, ticket: dict) -> dict:
        """
        Validate if the assigned tag matches the department routing rules in the policy.
        Return the validation result to support Human-in-the-loop when routing is invalid
        or confidence_score is low.
        """
        tags = ticket.get("tags", [])
        email_content = ticket.get("email_content", "").lower()
        department_rules = self.policy.get("department_rules", {})
        priority_mapping = self.policy.get("sentiment_thresholds", {}).get(
            "priority_mapping", {}
        )
        confidence_score = ticket.get("confidence_score")
        low_confidence = (
            confidence_score is not None and confidence_score < CONFIDENCE_THRESHOLD
        )

        routing_results = []
        for tag in tags:
            if tag not in department_rules:
                routing_results.append(
                    {
                        "tag": tag,
                        "department": None,
                        "valid": False,
                        "matched_keywords": [],
                    }
                )
                continue

            matched_keywords = [
                keyword
                for keyword in department_rules[tag]
                if self._keyword_matches(email_content, keyword)
            ]
            routing_results.append(
                {
                    "tag": tag,
                    "department": tag,
                    "valid": len(matched_keywords) > 0,
                    "matched_keywords": matched_keywords,
                }
            )

        # Route on the ML category; keyword tags only cross-check it (no tags = no conflict)
        category = ticket.get("category")
        known_category = category in department_rules
        is_valid = known_category and (not tags or category in tags)

        return {
            "valid": is_valid,
            "requires_human_review": not is_valid or low_confidence,
            "low_confidence": low_confidence,
            "routing": routing_results,
            "recommended_department": category if known_category else None,
            "priority": priority_mapping.get(ticket.get("sentiment"), "Medium"),
        }

    def classify_category(self, text: str) -> tuple[str, float, dict]:
        """
        ML-based Classification: calibrated linear SVM with TF-IDF.
        """
        return self.classifier.predict(text)

    def reply_examples(self, ticket: dict, n: int = 3) -> str:
        """RAG: phản hồi mẫu của nhân viên cho các vé tương tự, định dạng sẵn để chèn vào prompt."""
        try:
            from modules.vector_store import find_similar
            hits = find_similar(
                [ticket.get("email_content", "")], n=n,
                where={"$and": [{"source": "dataset"}, {"response": {"$ne": ""}}]},
                exclude_ids=[ticket.get("id")],
            )[0]
        except Exception as e:
            print(f"⚠️ Không lấy được phản hồi mẫu từ Vector DB: {e}")
            return ""
        if not hits:
            return ""
        lines = ["\nPhản hồi mẫu của nhân viên cho các yêu cầu tương tự (chỉ để tham khảo nội dung xử lý):"]
        for i, h in enumerate(hits, 1):
            lines.append(f"{i}. Khách: \"{h['text']}\"\n   Nhân viên: \"{h['response'][:600]}\"")
        return "\n".join(lines) + "\n"

    def generate_email_draft(self, ticket: dict, stream: bool = False):
        """
        AI-based Email Draft Generation using LLM API.
        """
        from modules.llm_client import chat_completion, chat_completion_stream

        category = ticket.get("category", "General")
        ticket_id = ticket.get("id", "N/A")
        customer_name = ticket.get("customer", "Customer")
        email_content = ticket.get("email_content", "")
        email_content_snippet = email_content[:50]
        sentiment = ticket.get("sentiment", "Neutral")
        priority = ticket.get("priority", "Medium")
        examples = self.reply_examples(ticket)

        prompt = f"""Dựa trên các thông tin sau:
- Category: {category}
- Ticket ID: {ticket_id}
- Customer: {customer_name}
- Sentiment: {sentiment}
- Priority: {priority}
- Nội dung khách hàng: "{email_content}"
{examples}
Hãy viết một email phản hồi khách hàng bằng tiếng Anh, tuân thủ đúng định dạng mẫu dưới đây:

SUBJECT: Update on your {category} request - Ref: {ticket_id}

Hi {customer_name},

[Empathy_Sentence]

We've received your request: "{email_content_snippet}..." 
Our {category} team is already looking into this. 

[Action_Plan]

Best regards,
The Support Team

Yêu cầu:
1. Thay thế [Empathy_Sentence] bằng một câu đồng cảm phù hợp (Negative: hối lỗi, Neutral: chuyên nghiệp).
2. Thay thế [Action_Plan] bằng các bước xử lý cụ thể cho vấn đề này (dựa vào phản hồi mẫu nếu có, ngắn gọn, không dùng placeholder dạng {{{{...}}}}) và một câu cam kết thời gian phản hồi (High: trong 2 hours, Medium/Low: trong 24 hours).

Chỉ xuất ra nội dung email hoàn chỉnh, không giải thích gì thêm:
"""

        try:
            sys_prompt = "Bạn là một trợ lý AI đóng vai trò chuyên gia CSKH (Customer Support)."
            if stream:
                return chat_completion_stream(prompt, system_prompt=sys_prompt)
            else:
                return chat_completion(prompt, system_prompt=sys_prompt)
        except Exception as e:
            error_msg = f"Lỗi gọi LLM API khi tạo email draft: {e}"
            raise RuntimeError(error_msg)

    def process_ticket(self, ticket: dict, prediction: tuple[str, float, dict] | None = None) -> dict:
        """Analyze sentiment, assign tags, classify category and validate routing."""
        email_content = ticket.get("email_content", "")

        processed = ticket.copy()

        predicted_category, confidence, prob_dict = prediction or self.classify_category(email_content)
        processed["confidence_score"] = round(confidence, 4)
        processed["category_probabilities"] = prob_dict

        processed["category"] = predicted_category
        
        # Sửa lỗi hiển thị "(Changed from nan)" trên UI
        orig_cat = ticket.get("category", "")
        if pd.isna(orig_cat):
            orig_cat = ""
        processed["original_csv_category"] = orig_cat
        
        processed["sentiment"] = self.analyze_sentiment(email_content)
        processed["tags"] = self.assign_tags(email_content)
        processed["routing_validation"] = self.validate_routing(processed)
        processed["priority"] = processed["routing_validation"]["priority"]
        
        return processed

    def process_all(self) -> list[dict]:
        """Processes all loaded tickets, classifying them in one batch."""
        if not self.tickets:
            self.load_tickets()

        return self.process_many(self.tickets)

    def process_many(self, tickets: list[dict]) -> list[dict]:
        """Classify tickets in one batch; low-confidence ones may follow a similar human correction."""
        if not tickets:
            return []

        texts = [t.get("email_content") if isinstance(t.get("email_content"), str) else "" for t in tickets]
        predictions = self.classifier.predict_many(texts)
        matches = self._follow_similar_corrections(texts, [t.get("id") for t in tickets], predictions)

        processed = []
        for i, (ticket, prediction) in enumerate(zip(tickets, predictions)):
            p = self.process_ticket(ticket, prediction)
            if i in matches:
                p["routed_by_correction"] = matches[i]
            processed.append(p)
        return processed

    def _follow_similar_corrections(self, texts: list[str], ids: list, predictions: list) -> dict[int, dict]:
        """
        Vé có độ tự tin thấp đi theo chỉnh sửa của nhân viên cho vé rất giống nó (Vector DB),
        để một chỉnh sửa có hiệu lực ngay cho các vé tương tự, không cần chờ train lại.
        Sửa predictions tại chỗ; trả về {index: thông tin vé chỉnh sửa khớp}.
        """
        from modules.feedback_manager import FeedbackManager

        low = [i for i, p in enumerate(predictions) if p[1] < CONFIDENCE_THRESHOLD]
        corrections = FeedbackManager(self.classifier.corrections_path).latest_by_ticket().values()
        if not low or not any(c.get("email_content") for c in corrections):
            return {}  # không cần nạp mô hình embedding

        try:
            from modules.vector_store import find_similar
            hits = find_similar([texts[i] for i in low], n=1, where={"source": "human_correction"},
                                exclude_ids=[ids[i] for i in low])
        except Exception as e:
            print(f"⚠️ Không tra được chỉnh sửa tương tự trong Vector DB: {e}")
            return {}

        matches = {}
        for i, h in zip(low, hits):
            if h and h[0]["similarity"] >= CORRECTION_MATCH_SIMILARITY:
                best = h[0]
                predictions[i] = (best["category"], best["similarity"], predictions[i][2])
                matches[i] = {"ticket_id": best["id"].removeprefix("correction_"), "similarity": best["similarity"]}
        return matches

    @staticmethod
    def _keyword_matches(text: str, keyword: str) -> bool:
        """Check if keyword appears in text (supports multi-word phrases)."""
        return bool(re.search(rf"\b{re.escape(keyword)}\b", text))

_training_lock = threading.Lock()


def is_training() -> bool:
    return _training_lock.locked()


def train_new_model_background(file_path: str):
    """Chạy background để train mô hình mới không gây treo UI. Chỉ 1 lần train tại một thời điểm."""
    if not _training_lock.acquire(blocking=False):
        print("⚠️ Một lần train khác đang chạy, bỏ qua yêu cầu này.")
        return
    try:
        print(f"🚀 Background Training started for data: {file_path}")
        MLClassifier(training_data_path=file_path, force_retrain=True)
        print("✅ Background Training completed successfully!")
    except Exception as e:
        print(f"❌ Background Training failed: {e}")
    finally:
        _training_lock.release()