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
* **Intelligent Categorization**: Uses `LogisticRegression` with TF-IDF to map emails to 5 departments.
* **Precision Confidence Scoring**: Features real-time metrics using `predict_proba()`.
* **Dynamic Prioritization**: Automated sentiment/keyword analysis to flag "High Priority" items.
* **Human-in-the-Loop (HITL)**: Agent feedback loops to override routing and retrain the model.

## 🔄 The Pipeline
1. **Ingestion**: Raw ticket data is loaded.
2. **ML Analysis**: Text is processed via TF-IDF and classified.
3. **Policy Validation**: Rules are applied from `company_policy.json`.
4. **Confidence Check**: Probabilities < 0.6 flag the ticket for **Human Intervention**.
5. **Review & Draft**: Agents use the **Streamlit UI** to verify or draft responses.



## 🏗 Project Structure
* `modules/`: Core logic (`processor.py`, `llm_client.py`, `orchestrator.py`).
* `data/`: Datasets, models, and `company_policy.json`.
* `ui/`: Streamlit-based agent dashboard.

## 🛠 Getting Started
1. **Environment Setup**: `pip install -r requirements.txt`.
2. **Configure**: Tweak routing rules in `data/company_policy.json`.
3. **Launch Engine**: `python modules/orchestrator.py`
4. **Launch UI**: `streamlit run ui.py`
5. **Calibrate**: Adjust `CONFIDENCE_THRESHOLD` in `modules/processor.py`.

---
*If you found this helpful or have suggestions, please open an issue or start a discussion!*
