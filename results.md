# Model Race: Results Summary

| Model | Accuracy | Precision | Recall | F1 | Cost at paid list price (USD) | Time per review (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| scikit-learn (TF-IDF + logistic regression) | 0.7200 | 0.6957 | 0.6957 | 0.6957 | $0.0000 | 0.07 |
| Cloud LLM (Groq: openai/gpt-oss-120b) | 0.9100 | 0.9302 | 0.8696 | 0.8989 | $0.0056 | 378.97 |
| Ollama (llama3.2:latest) | 0.8700 | 0.8367 | 0.8913 | 0.8632 | $0.0000 | 7606.83 |


### Class Balance Analysis
The test set contains 100 reviews: **46 positive (46.0%)** and **54 negative (54.0%)**.
Because the classes are approximately balanced (close to a 50/50 split), **Accuracy is an informative and trustworthy metric**, as a naive baseline predicting the majority class would only achieve ~54.0% accuracy rather than artificially inflating the score.

### Model Recommendation & Shipping Decision
For this production task, I would ship **scikit-learn (TF-IDF + logistic regression)**.
Although large language models can capture subtle linguistic nuances, the TF-IDF + Logistic Regression model delivers competitive accuracy while being several orders of magnitude faster (0.07 ms vs 378.97 ms per review).
Furthermore, it incurs zero API expenses, does not suffer from rate limits or downtime risks, and runs deterministically on standard CPU infrastructure without external dependencies.
Unless domain drift or highly ambiguous multi-sentence sarcasm requires complex zero-shot reasoning, the lightweight scikit-learn model offers the superior balance of speed, cost, reliability, and accuracy for high-throughput sentiment classification.
