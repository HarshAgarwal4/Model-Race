# Model Race: Small ML Model vs Cloud LLM vs Local LLM

A benchmarking project comparing the performance, latency, and cost of a lightweight Machine Learning model (**scikit-learn Logistic Regression**) against a Cloud LLM (**Groq: `openai/gpt-oss-120b`**) and a Local LLM (**Ollama: `llama3.2:latest`**) on binary sentiment classification.

---

## Project Overview

When building production NLP systems, developers frequently jump to Large Language Models without measuring trade-offs. The goal of this project is to benchmark:
1. **scikit-learn (TF-IDF + Logistic Regression)**: Ultra-fast, deterministic CPU inference with zero API costs.
2. **Cloud LLM (Groq)**: High-accuracy hosted cloud inference via structured outputs.
3. **Local LLM (Ollama)**: Private, self-hosted LLM running locally with zero API fee.

---

## Dataset Structure

The data directory (`data/`) contains:

```
data/
├── review.csv          # Raw scraped/collected dataset
└── cleaned_review.csv  # Preprocessed, deduplicated & encoded dataset
```

### Format & Schema:
- **`review.csv`**:
  - `review` *(string)*: Raw text of the movie/product review.
  - `sentiment` *(string)*: Review sentiment (`positive` or `negative`).
- **`cleaned_review.csv`**:
  - `review` *(string)*: Cleaned, whitespace-trimmed review string.
  - `sentiment` *(integer)*: Binary encoded (`1` = positive, `0` = negative). Duplicates and nulls removed.

---

## Prerequisites & Environment Setup

This project is managed using **[uv](https://docs.astral.sh/uv/)**.

### 1. Install dependencies with `uv`
```bash
uv sync
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and provide your API keys:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Ensure Ollama is Running (Local LLM)
Make sure Ollama is installed and the model is pulled locally:
```bash
ollama pull llama3.2:latest
ollama serve
```

---

## Steps to Run the Scripts

Always run commands via `uv run` to ensure execution inside the virtual environment:

### Step 1: Data Cleaning (`cleaning/clean.py`)
Cleans raw reviews, drops missing values/duplicates, and encodes labels to `0` and `1`:
```bash
uv run python cleaning/clean.py
```
*Output*: Generates `data/cleaned_review.csv`.

---

### Step 2: Train the ML Model (`models/model.py`)
Splits the dataset, fits the TF-IDF vectorizer on training data, trains the Logistic Regression classifier, evaluates metrics, and exports the serialized model:
```bash
uv run python models/model.py
```
*Output*: Saves model weights and vectorizer to `models/sentiment_model.pkl`.

---

### Step 3: Interactive Inference (`predict.py`)
Loads the trained model pipeline and predicts sentiment interactively for user-entered text:
```bash
uv run python predict.py
```
*Example Prompt*:
```
Enter your review: The cinematography was breathtaking and acting was superb!
Predicted Sentiment: positive
```

---

### Step 4: Run the Model Race (`race.py`)
Executes the race across scikit-learn, Groq, and Ollama, computing Accuracy, Precision, Recall, F1, Latency per review (ms), and Cost (USD):
```bash
uv run python race.py
```
*Output*: Prints the comparison table in the terminal and updates `results.md`.

---

## Batch Size Optimization & Troubleshooting

Inside `race.py`:
```python
BATCH_SIZE = 10
```
- By default, reviews are sent to Groq and Ollama in batches of 10 reviews.
- **Tip**: If you encounter **rate limits (HTTP 429)** on Groq, **context length limits**, or **timeouts/out-of-memory errors** on Ollama, **reduce the batch size to 5**:
  ```python
  BATCH_SIZE = 5
  ```
- Smaller batches significantly lower token load per structured JSON request, preventing truncated responses and ensuring high stability.

---

## Benchmark Results Summary

Results on the 100-review test set (from `results.md`):

| Model | Accuracy | Precision | Recall | F1 | Cost at paid list price (USD) | Time per review (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **scikit-learn (TF-IDF + Logistic Regression)** | 0.7400 | 0.8333 | 0.6000 | 0.6977 | **$0.0000** | **0.66 ms** |
| **Cloud LLM (Groq: openai/gpt-oss-120b)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | $0.0054 | 718.24 ms |
| **Local LLM (Ollama: llama3.2:latest)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **$0.0000** | 1274.58 ms |

### Key Takeaway:
- **scikit-learn** is **>1000x faster** (~0.66 ms vs 718 ms) and costs \$0.00, making it ideal for high-throughput, low-latency production pipelines.
- **LLMs** deliver perfect accuracy on nuanced or sarcastic reviews at the cost of higher latency and token pricing.
