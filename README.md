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
* **Intelligent Categorization**: A calibrated linear SVM (`LinearSVC` + `CalibratedClassifierCV`) on TF-IDF 1–2-grams routes tickets to seven departments: *Account Management, Billing, General Inquiry, Logistics, Returns, Sales and Technical.* Vietnamese text is word-segmented with `underthesea`.
* **Precision Confidence Scoring**: Every prediction carries a calibrated probability. Tickets below the `0.65` confidence threshold, or where the department keywords in `company_policy.json` contradict the model, go to the **Human Intervention Panel**.
* **Dynamic Prioritization**: Keyword-based sentiment analysis flags critical tickets (e.g. legal threats) as High priority.
* **Human-in-the-Loop (HITL)**: Agents correct the department from a dropdown. Corrections apply immediately in the UI and are merged into the training data on the next retrain.
* **Semi-supervised Training & Versioning**: Unlabelled tickets are labelled by the SVM, or by the LLM when the SVM is unsure (max 20 calls, answers validated against known departments). Each training run is saved as a version in `models/`, with accuracy measured on a held-out 20% of real labels.
* **AI Reply Drafts**: An OpenAI-compatible LLM (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`) drafts customer replies.

## 📊 Data
`data/tickets.csv` holds ~3,700 labelled tickets (~520 per department): Bitext tickets from the [Bitext customer support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), 250 generated English/Vietnamese Technical tickets (bugs, crashes, slow loading, errors) that Bitext lacks, and 210 original English/Vietnamese tickets. To rebuild it, download the Bitext CSV to `data/bitext/bitext_27k.csv` and run `python convert_bitext.py`. The intent → department mapping and the Technical templates are at the top of that script.

Known gaps: Bitext and the Technical tickets are template-based, so held-out accuracy (~99%) overstates real-world accuracy, and Vietnamese is thin outside Technical.

## 🛠 Getting Started
1. **Environment Setup**: `pip install -r requirements.txt`.
2. **Configure**: Edit departments and keywords in `data/company_policy.json`, and set the `LLM_*` environment variables for reply drafts and auto-labelling.
3. **Launch**: `streamlit run ui.py`. The first start trains a model if none exists.
4. **Retrain**: In the Admin Dashboard tab, upload a new dataset or click **Retrain with corrections**.
5. **Calibrate**: Adjust `CONFIDENCE_THRESHOLD` in `modules/processor.py` to balance automation against human review volume.
6. **Check**: `python test_routing.py` runs the routing self-check.