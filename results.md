# Model Race: Results

| Model | Accuracy | Precision | Recall | F1 | Cost at paid list price (USD) | Time per review (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| scikit-learn (TF-IDF + logistic regression) | 0.7400 | 0.8333 | 0.6000 | 0.6977 | $0.0000 | 0.66 ms |
| Cloud LLM (Groq: openai/gpt-oss-120b) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | $0.0054 | 718.24 ms |
| Local LLM (Ollama: llama3.2:latest) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | $0.0000 | 1274.58 ms |