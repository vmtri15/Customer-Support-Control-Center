# AI-Driven Ticket Routing System

## Overview
The **AI-Driven Ticket Routing System** is an intelligent automation pipeline designed to modernize customer support. By leveraging Natural Language Processing (NLP) and a dynamic business rules engine, this system eliminates manual triage, reduces response times, and ensures that critical customer issues never fall through the cracks.

## Project Goal
To replace inefficient, manual ticket routing with an automated, AI-driven pipeline, thereby eliminating human error and significantly decreasing ticket resolution latency.

## Expected Business Results
* **For Support Agents**: A drastic reduction in manual triage time, allowing them to focus on resolving complex issues rather than sorting them.
* **For Customer Success**: Higher Customer Satisfaction (CSAT) scores driven by faster, more accurate ticket routing.
* **For Management**: Optimized operational costs through increased efficiency and reduced overhead in the support department.

## 🌟 Key Features
* **Intelligent Categorization**: Uses a trained `LogisticRegression` model with TF-IDF vectorization to map incoming emails to five distinct departments: *Account Management, Billing, General Inquiry, Logistics, and Technical.*
* **Precision Confidence Scoring**: Features real-time confidence metrics using `predict_proba()`. Tickets failing to meet the `0.6` confidence threshold are automatically diverted to a **Human Intervention Panel** for expert oversight.
* **Dynamic Prioritization**: Performs automated sentiment and keyword analysis to flag "High Priority" items—like legal threats or crushed packages—so they hit the front of the queue.
* **Human-in-the-Loop (HITL)**: Built-in agent feedback loops allow support staff to override AI routing, ensuring the model continuously learns from real-world corrections.

## 🛠 Getting Started
1. **Environment Setup**: Install dependencies via `pip install -r requirements.txt`.
2. **Configure**: Tweak your routing rules in `data/company_policy.json` (categories and keywords).
3. **Launch**: Run the orchestration module to begin automated processing.
4. **Calibrate**: Adjust `CONFIDENCE_THRESHOLD` in `modules/processor.py` to balance the trade-off between AI automation and human review volume.
