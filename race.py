import asyncio
import time

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models.llm import generate_response
from predict import model_call

# ------------------------------------------------------------------
# Test Reviews & Labels (Populate with your 100 reviews and labels)
# ------------------------------------------------------------------

df = pd.read_csv('data/test_reviews.csv')

reviews: list[str] = df['review'].to_list()
y_true: list[int] = df['sentiment'].astype(int).to_list()

BATCH_SIZE_GROQ = 10
BATCH_SIZE_OLLAMA = 10

def evaluate(y_test: list[int], y_pred: list[int]) -> dict:
    """Calculates accuracy, precision, recall, and f1 score."""
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

def calculate_cost(provider: str, input_tokes=0 , output_tokens=0) -> float:
    """Calculates estimated cost in USD at paid list price."""
    if provider in ("scikit-learn", "Ollama"):
        return 0.0000
    elif provider == "Groq":
        return (input_tokes / 1_000_000) * 0.59 + (output_tokens / 1_000_000) * 0.79
    return 0.0000

def save_results(content: str):
    with open("results.md", "w", encoding="utf-8") as f:
        f.write(content)

async def main():
    try:
        if not reviews or not y_true:
            print("[!] reviews or y_true is empty. Please populate the 'reviews' and 'y_true' lists to run the race.")
            return
        results = []
        n_reviews = len(reviews)

        # 1. scikit-learn (using model_call)
        print("\n[1/3] Running scikit-learn model via model_call...")
        start_time = time.perf_counter()
        y_pred_model = [await model_call(r) for r in reviews]
        time_model = time.perf_counter() - start_time
        metrics_model = evaluate(y_true, y_pred_model)
        time_per_rev_model = (time_model / n_reviews) * 1000
        print(metrics_model)
        results.append({
            "name": "scikit-learn (TF-IDF + logistic regression)",
            "accuracy": metrics_model["accuracy"],
            "precision": metrics_model["precision"],
            "recall": metrics_model["recall"],
            "f1": metrics_model["f1"],
            "cost": calculate_cost("scikit-learn"),
            "time_per_rev": time_per_rev_model,
        })

        # 2. Cloud LLM (Groq) in batches of 10
        print("\n[2/3] Running Cloud LLM (Groq: openai/gpt-oss-120b) in batches of 10...")
        start_time = time.perf_counter()
        y_pred_groq = []
        input_tokens = 0
        output_tokens = 0

        for batch_no,i in enumerate(range(0, n_reviews, BATCH_SIZE_GROQ), start=1):
            batch = reviews[i:i + BATCH_SIZE_GROQ]
            print(f"Batch {batch_no} processing ...")
            res = await generate_response(batch, "Groq")
            input_tokens += res["metadata"]["input_tokens"]
            output_tokens += res["metadata"]["output_tokens"]

            for prediction in res["response"]:
                y_pred_groq.append(
                    1 if prediction.sentiment.lower() == "positive" else 0
                )
        time_groq = time.perf_counter() - start_time
        metrics_groq = evaluate(y_true, y_pred_groq)
        time_per_rev_groq = (time_groq / n_reviews) * 1000
        print(metrics_groq)
        results.append({
            "name": "Cloud LLM (Groq: openai/gpt-oss-120b)",
            "accuracy": metrics_groq["accuracy"],
            "precision": metrics_groq["precision"],
            "recall": metrics_groq["recall"],
            "f1": metrics_groq["f1"],
            "cost": calculate_cost("Groq", input_tokes=input_tokens , output_tokens=output_tokens),
            "time_per_rev": time_per_rev_groq,
        })

        # 3. Local LLM (Ollama) in batches of 10
        print("\n[3/3] Running Local LLM (Ollama: llama3.2:latest) in batches of 10...")
        start_time = time.perf_counter()
        y_pred_ollama = []
        input_tokens = 0
        output_tokens = 0
        for batch_no,i in enumerate(range(0, n_reviews, BATCH_SIZE_OLLAMA),start=1):
            batch = reviews[i:i + BATCH_SIZE_OLLAMA]
            print(f"Batch {batch_no} processing ...")
            res = await generate_response(batch, "Ollama")
            input_tokens += res["metadata"]["input_tokens"]
            output_tokens += res["metadata"]["output_tokens"]
            for prediction in res["response"]:
                y_pred_ollama.append(
                    1 if prediction.sentiment.lower() == "positive" else 0
                )
        time_ollama = time.perf_counter() - start_time
        metrics_ollama = evaluate(y_true, y_pred_ollama)
        time_per_rev_ollama = (time_ollama / n_reviews) * 1000
        print(metrics_ollama)
        results.append({
            "name": "Local LLM (Ollama: llama3.2:latest)",
            "accuracy": metrics_ollama["accuracy"],
            "precision": metrics_ollama["precision"],
            "recall": metrics_ollama["recall"],
            "f1": metrics_ollama["f1"],
            "cost": calculate_cost("Ollama", input_tokes=input_tokens , output_tokens=output_tokens),
            "time_per_rev": time_per_rev_ollama,
        })

        # Output Comparison Table
        table_lines = [
            "| Model | Accuracy | Precision | Recall | F1 | Cost at paid list price (USD) | Time per review (ms) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in results:
            table_lines.append(
                f"| {r['name']} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | ${r['cost']:.4f} | {r['time_per_rev']:.2f} ms |"
            )

        table_md = "\n".join(table_lines)
        print("\n" + "=" * 60)
        print(" MODEL RACE RESULTS")
        print("=" * 60)
        print(table_md)

        await asyncio.to_thread(
            save_results,
            "# Model Race: Results\n\n" + table_md + "\n"
        )
        print("\n[✓] Results saved to results.md")
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    asyncio.run(main())
