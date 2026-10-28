# AI-Driven Ticket Routing System

## Overview
This AI-Driven Ticket Routing System modernizes support by replacing manual triage with an automated NLP pipeline. It categorizes tickets using machine learning and routes them based on real-time confidence. Ambiguous cases move to a Human Intervention Panel, ensuring operational efficiency, lower latency, and improved CSAT scores for your support team.

## Project Goal
To replace inefficient, manual ticket routing with an automated, AI-driven pipeline, thereby eliminating human error and significantly decreasing ticket resolution latency.

## Expected Business Results
* **For Support Agents**: A drastic reduction in manual triage time.
* **For Customer Success**: Higher CSAT scores driven by faster, accurate routing.
* **For Management**: Optimized operational costs through increased efficiency.

## 🌟 Key Features
* **Intelligent Categorization**: A calibrated linear SVM (`LinearSVC` + `CalibratedClassifierCV`) on TF-IDF 1–2-grams routes tickets to seven departments: *Account Management, Billing, General Inquiry, Logistics, Returns, Sales and Technical.* Vietnamese text is word-segmented with `underthesea`.
* **Precision Confidence Scoring**: Every prediction carries a calibrated probability. Tickets below the `0.65` confidence threshold, or where the department keywords in `company_policy.json` contradict the model, go to the **Human Intervention Panel**.
* **Dynamic Prioritization**: Keyword-based sentiment analysis flags critical tickets (e.g. legal threats) as High priority.
* **Human-in-the-Loop (HITL)**: Agents correct the department from a dropdown. Corrections apply immediately in the UI and are merged into the training data on the next retrain.
* **Semi-supervised Training & Versioning**: Unlabelled tickets are labelled by the SVM, or by the LLM when the SVM is unsure (max 20 calls, answers validated against known departments). Each training run is saved as a version in `models/`, with accuracy measured on a held-out 20% of real labels.
* **AI Reply Drafts (RAG)**: An OpenAI-compatible LLM (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`) drafts customer replies, using the agent replies of the 3 most similar past tickets as examples. Agents edit the draft and send it: it is added to the ticket's conversation and, when SMTP is configured and an address is given, emailed to the customer.
* **AI Solver (auto-resolution)**: When a customer submits a request the classifier is sure about (≥85% confidence, no review flag, not High priority, not Negative), the LLM answers it straight away, grounded only in the agent replies of similar past tickets; it hands over (`ESCALATE`) when those don't cover the request. The customer clicks **Yes, this solved it** (ticket resolved) or **I still need help** (back to the agent queue). Analytics shows the share of AI answers customers accepted.
* **Response-time targets & metrics**: Each priority has a first-reply and resolution target (High 1 h / 8 h, Medium 4 h / 24 h, Low 24 h / 72 h; `SLA_HOURS` in `modules/store.py`). The Agent Console shows what's due next, warns on overdue tickets and filters to them; nothing is due from agents while a ticket waits on the customer. Analytics reports median first-reply and resolution times, on-time rates by priority, customer satisfaction (1–5 stars, rated in the Help Center after resolution), AI solved rate and daily volume.
* **Vector DB (ChromaDB)**: Tickets are embedded with `all-MiniLM-L6-v2`. Agents can **Show similar past tickets**, and low-confidence tickets that closely match (≥85% similar) a human correction are routed like it immediately, before any retrain. The index rebuilds automatically when `data/tickets.csv` changes.

## 🔄 The Pipeline
1. **Ingestion**: Customers submit requests in the **Help Center**, and the demo tickets from `data/tickets.csv` are loaded on first run. Everything is stored in SQLite (`data/support.db`).
2. **ML Analysis**: Text is preprocessed, vectorized with TF-IDF and classified by the SVM.
3. **Policy Validation**: Department keywords from `company_policy.json` cross-check the predicted department and set priority.
4. **Confidence Check**: Confidence < 0.65 or a keyword conflict flags the ticket for **Human Intervention**. Confident, calm, non-urgent requests from the Help Center go to the **AI Solver** first.
5. **Work the queue**: In the **Agent Console**, agents pick tickets from priority-sorted queues, assign them, reply in the conversation thread (AI drafts available), change status (Open → In Progress → Waiting on Customer → Resolved) and confirm or correct the department. Every change is written to the ticket's activity log.
6. **Customer follow-up**: In the Help Center, customers track a request with its ticket number and email, read the agents' replies and answer. An answer reopens a ticket that was waiting on them or resolved.
7. **Learn**: Corrections are merged into the training data on the next retrain.

## 📊 Data
`data/tickets.csv` holds ~3,700 labelled tickets (~520 per department): Bitext tickets from the [Bitext customer support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), 250 generated English/Vietnamese Technical tickets (bugs, crashes, slow loading, errors) that Bitext lacks, and 210 original English/Vietnamese tickets. To rebuild it, download the Bitext CSV to `data/bitext/bitext_27k.csv` and run `python convert_bitext.py`. The intent → department mapping and the Technical templates are at the top of that script.

Known gaps: Bitext and the Technical tickets are template-based, so held-out accuracy (~99%) overstates real-world accuracy, and Vietnamese is thin outside Technical.

## 🏗 Project Structure
* `modules/`: Core logic (`processor.py`, `feedback_manager.py`, `llm_client.py`, `vector_store.py`, `embeddings.py`).
* `data/`: Datasets, `company_policy.json` and `corrections.json`.
* `models/`: Versioned trained models and `metadata.json`.
* `ui.py` + `app_pages/`: Streamlit multipage app (Help Center for customers; Agent Console, Analytics and Admin for the support team).
* `modules/store.py`: SQLite ticket store (tickets, messages, events, agents).
* `convert_bitext.py`: Builds `data/tickets.csv` (including Bitext agent replies for RAG).
* `build_vectordb.py`: Rebuilds the ChromaDB index manually.

## 🛠 Getting Started
1. **Environment Setup**: `pip install -r requirements.txt`.
2. **Configure**: Edit departments and keywords in `data/company_policy.json`, and set the `LLM_*` environment variables for reply drafts and auto-labelling. To send replies by email, also set the SMTP variables (works with Gmail/Outlook app passwords, SendGrid, Mailgun or any SMTP relay):
   ```bash
   export SMTP_HOST=smtp.gmail.com SMTP_PORT=587 SMTP_FROM=support@yourcompany.com
   export SMTP_USER=support@yourcompany.com SMTP_PASSWORD='your-app-password'
   ```
   Port 465 uses SSL; other ports use STARTTLS unless `SMTP_STARTTLS=false`.
3. **Launch UI**: `streamlit run ui.py`. The first start trains a model if none exists and fills `data/support.db` with the triaged demo tickets, given a generated 30-day history (agent replies from the dataset, mostly on time, some late, some rated) so the metrics have something to show. Delete the file to reset the demo.
4. **Retrain**: In the Admin Dashboard tab, upload a new dataset or click **Retrain with corrections**.
5. **Calibrate**: Adjust `CONFIDENCE_THRESHOLD` (human review) and `AUTO_SOLVE_CONFIDENCE` (AI Solver) in `modules/processor.py` to balance automation against human workload.
6. **Check**: `python test_routing.py` (routing and sentiment), `python test_store.py` (ticket store) and `python test_features.py` (training, feedback, Vector DB, email; runs in a temp folder).

---
*If you found this helpful or have suggestions, please open an issue or start a discussion!*
