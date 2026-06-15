import json
import re
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

CONFIDENCE_THRESHOLD = 0.65


class MLClassifier:
    """A One-layer ML model using TF-IDF and Logistic Regression for ticket classification."""
    def __init__(self, training_data_path: str = "data/tickets.csv"):
        import joblib
        
        self.model_path = "data/logreg_model.joblib"
        self.vectorizer_path = "data/tfidf_vectorizer.joblib"
        
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.is_trained = True
            print("✅ Loaded pre-trained ML classifier from disk.")
        else:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.model = LogisticRegression(max_iter=1000)
            self.is_trained = False
            self._train(training_data_path)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.vectorizer, self.vectorizer_path)

    def _train(self, data_path: str):
        try:
            df = pd.read_csv(data_path)
            if 'email_content' in df.columns and 'category' in df.columns:
                # Drop rows with empty categories for training
                df = df.dropna(subset=['category', 'email_content'])
                X = self.vectorizer.fit_transform(df['email_content'])
                y = df['category']
                self.model.fit(X, y)
                self.is_trained = True
            else:
                print("⚠️ Missing required columns for training ML Classifier.")
        except Exception as e:
            print(f"⚠️ Error training ML Classifier: {e}")

    def predict(self, text: str) -> tuple[str, float, dict]:
        """Predicts the category, returns confidence, and all probabilities."""
        if not self.is_trained or not text.strip():
            return "General Inquiry", 0.0, {}
            
        X_new = self.vectorizer.transform([text])
        
        # 1. Get the probabilities for all categories
        probs = self.model.predict_proba(X_new)[0]
        
        # 2. Identify the highest probability (the confidence)
        confidence = float(max(probs))
        
        # 3. Identify the predicted category
        prediction = self.model.predict(X_new)[0]

        # 4. Map probabilities to category names
        prob_dict = {str(self.model.classes_[i]): float(probs[i]) for i in range(len(probs))}
        
        return prediction, confidence, prob_dict


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

        if any(keyword in text_lower for keyword in critical_keywords):
            return "Negative"

        department_keywords = [
            keyword
            for keywords in self.policy.get("department_rules", {}).values()
            for keyword in keywords
        ]
        if any(self._keyword_matches(text_lower, keyword) for keyword in department_keywords):
            return "Negative"

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

        is_valid = bool(routing_results) and all(
            result["valid"] for result in routing_results
        )

        return {
            "valid": is_valid,
            "requires_human_review": not is_valid or low_confidence,
            "low_confidence": low_confidence,
            "routing": routing_results,
            "recommended_department": routing_results[0]["department"]
            if is_valid
            else None,
            "priority": priority_mapping.get(ticket.get("sentiment"), "Medium"),
        }

    def classify_category(self, text: str) -> tuple[str, float, dict]:
        """
        ML-based Classification: One-layer Neural Network (Logistic Regression) with TF-IDF.
        """
        return self.classifier.predict(text)

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

        prompt = f"""Dựa trên các thông tin sau:
- Category: {category}
- Ticket ID: {ticket_id}
- Customer: {customer_name}
- Sentiment: {sentiment}
- Priority: {priority}

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
2. Thay thế [Action_Plan] bằng một câu cam kết thời gian phản hồi (High: trong 2 hours, Medium/Low: trong 24 hours).

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

    def process_ticket(self, ticket: dict) -> dict:
        """Analyze sentiment, assign tags, classify category and validate routing."""
        email_content = ticket.get("email_content", "")

        processed = ticket.copy()
        
        predicted_category, confidence, prob_dict = self.classify_category(email_content)
        processed["confidence_score"] = round(confidence, 4)
        processed["category_probabilities"] = prob_dict

        processed["category"] = predicted_category
        processed["original_csv_category"] = ticket.get("category", "")
        
        processed["sentiment"] = self.analyze_sentiment(email_content)
        processed["tags"] = self.assign_tags(email_content)
        processed["routing_validation"] = self.validate_routing(processed)
        processed["priority"] = processed["routing_validation"]["priority"]
        
        return processed

    def process_all(self, max_workers: int = 5) -> list[dict]:
        """
        Processes all loaded tickets in parallel blocks to maximize local vLLM performance
        and prevent UI hanging/timeouts on large datasets.
        """
        if not self.tickets:
            self.load_tickets()
            
        if not self.tickets:
            return []
            
        # Use a ThreadPoolExecutor to handle multiple API/processing requests concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            processed_tickets = list(executor.map(self.process_ticket, self.tickets))
            
        return processed_tickets

    @staticmethod
    def _keyword_matches(text: str, keyword: str) -> bool:
        """Check if keyword appears in text (supports multi-word phrases)."""
        return bool(re.search(rf"\b{re.escape(keyword)}\b", text))