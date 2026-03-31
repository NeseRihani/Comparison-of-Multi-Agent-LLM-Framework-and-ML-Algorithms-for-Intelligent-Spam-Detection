# 🤖 Multi-Agent LLM Framework for Email Spam Detection

A hierarchical multi-agent system that uses **Llama 3.3 70B** (via Groq API) to detect email spam through three specialized sub-agents and an adaptive orchestrator — with **no training data required**.

Built on the **CEAS-08** benchmark dataset and compared against traditional ML models (Naive Bayes, Logistic Regression, SGD Classifier, Random Forest, XGBoost).

---

## 📌 Overview

Most spam detection systems either require large labeled training sets or rely on a single model making all decisions. This project takes a different approach: multiple agents independently analyze different aspects of each email (content, sender, URLs), and a central orchestrator combines their verdicts through majority voting.

The system achieves **95.27% accuracy** and **95.71% F1-score** on 3,915 emails — in a **zero-shot setting** with no fine-tuning.

---

## 🏗️ Architecture

```
                    ┌─────────────────────┐
                    │   Orchestrator Agent │
                    │   (Majority Voting)  │
                    └────────┬────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
 ┌────────▼────────┐ ┌───────▼───────┐ ┌───────▼───────┐
 │  Content Agent  │ │  Sender Agent │ │   URL Agent   │
 │  Body & Subject │ │  Domain &     │ │  Hyperlinks   │
 │  Analysis       │ │  Address      │ │  (conditional)│
 └─────────────────┘ └───────────────┘ └───────────────┘
```

- 📝 **Content Agent** — analyzes subject line and body for spam language: unsolicited offers, credential requests, urgency cues, obfuscated text, impersonation patterns.
- 📧 **Sender Agent** — evaluates sender address and subject consistency: typosquatted domains, free email providers used for official comms, randomly generated addresses, all-caps panic subjects.
- 🔗 **URL Agent** — activated only when URLs are present (`urls = 1`). Checks for shorteners, IP-based links, misspelled brand domains, suspicious TLDs, and sender-URL domain mismatches. Automatically returns safe when no URLs exist.
- 🧠 **Orchestrator Agent** — aggregates votes with an adaptive threshold:
  - URLs present → needs 2/3 suspicious votes → spam
  - No URLs → needs 1/2 suspicious votes → spam

---

## 📊 Results

### Overall Performance (n = 3,915)

| System | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Multi-Agent LLM (Llama 3.3 70B) | 95.27% | 96.90% | 94.55% | 95.71% |

**Confusion Matrix:**
```
TP: 2,065   FN: 119
FP:    66   TN: 1,665
```

### 📈 Incremental Evaluation

| n | Spam | Accuracy | Precision | Recall | F1-Score |
|---|------|----------|-----------|--------|----------|
| 100 | 54 | 94.00% | 94.44% | 94.44% | 94.44% |
| 200 | 109 | 95.00% | 95.41% | 95.41% | 95.41% |
| 500 | 280 | 96.00% | 97.46% | 95.36% | 96.40% |
| 1,000 | 568 | 96.00% | 97.14% | 95.77% | 96.45% |
| 3,915 | 2,184 | 95.27% | 96.90% | 94.55% | 95.71% |

### 🔬 Per-Agent Ablation Study (n = 3,915)

| Agent | Accuracy | Precision | Recall | F1-Score | TP | FP | FN |
|-------|----------|-----------|--------|----------|----|----|----|
| Content Agent | 87.40% | 98.40% | 78.70% | 87.40% | 1,718 | 28 | 466 |
| Sender Agent | 84.70% | 94.80% | 76.70% | 84.70% | 1,676 | 92 | 508 |
| URL Agent | 71.60% | 90.90% | 54.60% | 68.20% | 1,193 | 120 | 991 |
| **Multi-Agent** | **95.27%** | **96.90%** | **94.55%** | **95.71%** | **2,065** | **66** | **119** |

> Combining all three agents improves F1 by **+8.31 points** over the best individual agent.

### ⚖️ ML Comparison (n = 3,915, trained on same data)

| Model | Accuracy | F1-Score | Training Required |
|-------|----------|----------|-------------------|
| Naive Bayes | 96.20% | 96.00% | ✅ Yes |
| Logistic Regression | 98.20% | 98.20% | ✅ Yes |
| SGD Classifier | 98.50% | 98.50% | ✅ Yes |
| Random Forest | 97.50% | 97.40% | ✅ Yes |
| XGBoost | 96.80% | 96.80% | ✅ Yes |
| **Multi-Agent LLM** | **95.27%** | **95.71%** | **❌ No (zero-shot)** |

> The LLM system performs within **2.8 F1 points** of the best ML model — **without any training data**.

### 🎯 Adaptive Threshold Impact

| Threshold | Emails | Accuracy | Precision | Recall | F1 |
|-----------|--------|----------|-----------|--------|----|
| URL present (2/3) | 2,602 | 94.9% | 98.3% | 92.7% | 95.4% |
| No URL (1/2) | 1,313 | 96.0% | 94.1% | 98.5% | 96.3% |

---

## 📁 Dataset

**CEAS-08** — collected during the 2008 Spam Challenge.

| Property | Value |
|----------|-------|
| Total emails | 39,154 |
| Spam | 21,842 (55.8%) |
| Ham | 17,312 (44.2%) |
| Fields | sender, receiver, date, subject, body, urls (binary), label |

The `urls` field is a binary feature: `1` = body contains at least one URL, `0` = no URLs. During preprocessing, actual hyperlink strings are extracted from the body via regex and stored in `url_list` for the URL Agent.

> 📦 Source: Part of the curated phishing email dataset by Champa et al. (IEEE ICMI 2024), available on [Kaggle](https://www.kaggle.com/).

---

## 📂 Project Structure

```
├── prepare_and_clean_ceas.py    # Data cleaning and URL feature extraction
├── spam_detection.py            # Main multi-agent detection system
├── ml_comparison.py             # Traditional ML baseline experiments
├── llm_analysis.py              # LLM results analysis and incremental breakdown
├── parse_agent_report.py        # Per-agent vote extraction from report logs
├── cleaned_ceas.csv             # Preprocessed dataset (after running prepare script)
├── sample_checkpoint.csv        # Fixed evaluation sample (seed=42)
├── results_checkpoint.csv       # Agent verdicts saved incrementally
├── analysis_report_ceas_groq.txt  # Detailed per-email agent decision log
├── analysis_results_ceas_groq.csv # Final results CSV
├── ml_results.csv               # ML model performance across data sizes
├── agent_votes.csv              # Per-agent binary votes (parsed from log)
└── agent_analysis.csv           # Incremental per-agent metrics
```

---

## ⚙️ Setup

### Requirements

```bash
pip install pandas groq scikit-learn scipy xgboost
```

### 🔑 API Key

Get a free API key from [console.groq.com](https://console.groq.com) and add it to a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
# You can add more keys for faster processing (round-robin rotation)
GROQ_API_KEY_2=your_second_key_here
```

> ⚠️ Never hardcode API keys directly in `.py` files. Always use a `.env` file and make sure it is listed in `.gitignore`.

---

## 🚀 Usage

### Step 1 — Prepare the dataset

```bash
python prepare_and_clean_ceas.py
```

Cleans CEAS-08, removes duplicates, fills missing fields, and extracts URL strings from email bodies. Outputs `cleaned_ceas.csv`.

### Step 2 — Run the multi-agent system

```bash
python spam_detection.py
```

Analyzes emails using three LLM agents. Saves progress to `results_checkpoint.csv` after every email — if interrupted, it resumes automatically from where it left off.

To restart from scratch:

```python
RESET_CHECKPOINT = True  # in spam_detection.py
```

### Step 3 — Analyze LLM results

```bash
python llm_analysis.py
```

Generates incremental performance metrics (n=100, 200, 500, 1000, full) and error breakdown from the checkpoint file.

### Step 4 — Run ML baseline

```bash
python ml_comparison.py
```

Trains and evaluates Naive Bayes, Logistic Regression, SGD, Random Forest, and XGBoost at the same incremental data sizes.

### Step 5 — Per-agent analysis

```bash
python parse_agent_report.py
```

Parses the detailed log file and extracts each agent's individual decisions. Validates for duplicate or missing email IDs, then produces `agent_votes.csv` and `agent_analysis.csv`.

---

## 💡 Key Design Decisions

- **🎚️ Adaptive threshold** — The URL Agent is inactive for emails with no hyperlinks. Rather than always requiring 2/3 votes, the threshold drops to 1/2 when the URL Agent cannot contribute. This prevents URL-free spam from being systematically missed.

- **💾 Checkpoint system** — The Groq free tier has per-minute token limits. The checkpoint system saves every result to disk immediately, so long runs can be interrupted and resumed without losing progress.

- **🔄 Round-robin key rotation** — Multiple Groq API keys are rotated across requests. When a key hits its rate limit, the system switches to the next one immediately without waiting.

- **🧪 Zero-shot inference** — No CEAS-08 examples are shown to the model during evaluation. Agent behavior is entirely prompt-driven.

---

## ⚠️ Limitations

- **Rate limits** — The Groq free tier limits throughput. Processing 3,915 emails required multiple sessions across different days.
- **URL-free spam** — The URL Agent cannot contribute for emails without hyperlinks (~33% of the dataset). The adaptive threshold partially compensates, but these cases have higher false-negative rates.
- **Prompt sensitivity** — Agent decisions depend on prompt wording. No systematic prompt optimization was performed.

---

## 🎓 Academic Context

This project was developed as part of a conference paper submission. The multi-agent architecture and results are described in:

> **A Multi-Agent LLM Framework for Intelligent Email Spam Detection**

Dataset reference: Champa, A.I., Rabbi, M.F., and Zibran, M.F. — *Curated Datasets and Feature Analysis for Phishing Email Detection with Machine Learning*, IEEE ICMI 2024.

---

## 📄 License

**MIT License** — free to use, modify, and distribute with attribution.
