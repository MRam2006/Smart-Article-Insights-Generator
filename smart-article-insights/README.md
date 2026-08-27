# 📰 Smart Article Insights Generator

A local AI-powered application that summarizes articles and answers questions using Retrieval-Augmented Generation (RAG).

| Feature | Technology |
|---|---|
| LLM | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| UI | Gradio |
| Language | Python 3.12 |

---

## Project Structure

```
smart-article-insights/
├── app.py               # Main Gradio application
├── requirements.txt     # All dependencies (pinned)
├── Articles.csv         # Sample article dataset for RAG
├── README.md            # This file
├── .env.example         # Environment variable template
└── .gitignore           # Git ignore rules
```

---

## Quick Start

### 1. Prerequisites

- Python 3.12 installed ([python.org](https://www.python.org/downloads/))
- `pip` available in your PATH

Verify your Python version:

```powershell
python --version   # should print Python 3.12.x
```

---

### 2. Create a Virtual Environment

```powershell
# Navigate to the project folder
cd smart-article-insights

# Create the virtual environment
python -m venv .venv

# Activate it (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

> **Tip (Windows):** If you get an execution-policy error, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

### 3. Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏳ First install downloads PyTorch and Transformers (~2 GB). Subsequent installs use pip's cache.

---

### 4. Configure Environment (Optional)

```powershell
# Copy the example file
copy .env.example .env

# Edit .env only if you need a Hugging Face token or custom cache path.
# TinyLlama and all-MiniLM-L6-v2 are public — no token required.
```

---

### 5. Run the App

```powershell
python app.py
```

Open your browser at **http://127.0.0.1:7860**

---

## Phases

| Phase | Status | Description |
|---|---|---|
| 1 | ✅ Done | Project scaffold + Gradio UI shell |
| 2 | 🔜 Next | TinyLlama summarization |
| 3 | 🔜 Planned | RAG pipeline (FAISS + Q&A) |

---

## Notes

- All model inference runs **locally on CPU** by default; CUDA is used automatically if a compatible GPU is detected.
- Model weights are cached in `~/.cache/huggingface` after the first download.
- No data leaves your machine — fully offline after models are downloaded.
