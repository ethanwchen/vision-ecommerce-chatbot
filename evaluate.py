"""
evaluate.py

Evaluates the FAISS retrieval system using Recall@1, Recall@5, and Recall@10.

A synthetic test set is created by using each product's own name as the query;
the ground-truth is that the query should retrieve that same product.

Usage:
    python evaluate.py
"""

import random
from rag import retrieve, load_resources, get_products


def create_test_queries(products, n=200, seed=42):
    """
    Sample n products and use their name as the query.
    Ground truth: the query should retrieve the product with the same id.
    """
    random.seed(seed)
    sample = random.sample(products, min(n, len(products)))
    return [{"query": p["name"], "product_id": p["id"]} for p in sample if p["name"]]


def evaluate_recall(test_queries, k_values=(1, 5, 10)):
    max_k = max(k_values)
    hits  = {k: 0 for k in k_values}

    for item in test_queries:
        results = retrieve(query_text=item["query"], k=max_k)
        retrieved_ids = [r["id"] for r in results]
        for k in k_values:
            if item["product_id"] in retrieved_ids[:k]:
                hits[k] += 1

    n = len(test_queries)
    return {f"Recall@{k}": round(hits[k] / n, 4) for k in k_values}


def accuracy_at_1(test_queries):
    """Fraction of queries where the top result is the correct product."""
    correct = 0
    for item in test_queries:
        results = retrieve(query_text=item["query"], k=1)
        if results and results[0]["id"] == item["product_id"]:
            correct += 1
    return round(correct / len(test_queries), 4)


if __name__ == "__main__":
    print("Loading resources...")
    load_resources()
    products = get_products()

    print(f"Building test set from {len(products)} products...")
    test_queries = create_test_queries(products, n=200)
    print(f"  Test set size: {len(test_queries)}")

    print("Evaluating retrieval...")
    recall_metrics = evaluate_recall(test_queries, k_values=(1, 5, 10))
    acc1 = accuracy_at_1(test_queries)

    print("\n--- Retrieval Evaluation Results ---")
    print(f"Accuracy@1 : {acc1:.4f}")
    for metric, value in recall_metrics.items():
        print(f"{metric:12s}: {value:.4f}")
