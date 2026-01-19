import json
import os
from dotenv import load_dotenv
from retrieval.normal_retriever import normal_retriever
from vectorstore.set_up_collections import get_coarse_chunk_store

load_dotenv()

GROUND_TRUTH_FILE = os.getenv("EVALUATION_GROUND_TRUTH_FILE", "ground_truth.json")
RESULTS_FILE = os.getenv("COARSE_EVAL_RESULTS", "coarse_eval_results.json")
TOP_K = 10

def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["queries"]

def evidence_recall_at_k(retrieved, required_evidence, k):
    retrieved = retrieved[:k]

    retrieved_pairs = set()
    for doc in retrieved:
        for ch in doc["chunks"]:
            retrieved_pairs.add((ch["doc_id"], ch["chunk_id"]))

    required_pairs = set()
    for doc_id, chunk_ids in required_evidence.items():
        for cid in chunk_ids:
            required_pairs.add((doc_id, cid))

    covered = retrieved_pairs & required_pairs
    recall = len(covered) / len(required_pairs)

    return recall, covered, required_pairs


def and_hop_success(retrieved, required_evidence):
    for doc_id, chunk_ids in required_evidence.items():
        found = False
        for doc in retrieved:
            if doc["doc_id"] == doc_id:
                retrieved_chunk_ids = {c["chunk_id"] for c in doc["chunks"]}
                if any(cid in retrieved_chunk_ids for cid in chunk_ids):
                    found = True
        if not found:
            return 0
    return 1


if __name__ == "__main__":

    ground_truth = load_ground_truth(GROUND_TRUTH_FILE)
    chunk_store = get_coarse_chunk_store()

    all_results = []
    evidence_recalls = []
    hop_scores = []

    for q in ground_truth:
        query = q["query"]
        required_evidence = q["required_evidence"]

        retrieved = normal_retriever(
            query=query,
            chunk_store=chunk_store,
            top_k_chunks=50,
            final_top_k_docs=TOP_K
        )

        recall, covered, total = evidence_recall_at_k(retrieved, required_evidence, TOP_K)
        hop = and_hop_success(retrieved, required_evidence)

        evidence_recalls.append(recall)
        hop_scores.append(hop)

        all_results.append({
            "query_id": q["id"],
            "query": query,
            "evidence_recall_at_k": recall,
            "and_hop_success": hop,
            "required_evidence": list(total),
            "covered_evidence": list(covered),
            "retrieved_docs": [r["doc_id"] for r in retrieved]
        })

        print("=" * 80)
        print("Query:", query)
        print("Evidence Recall@{}: {:.3f}".format(TOP_K, recall))
        print("AND-Hop Success:", hop)
        print("Covered Evidence:", covered)
        print("=" * 80)

    avg_recall = sum(evidence_recalls) / len(evidence_recalls)
    avg_hop = sum(hop_scores) / len(hop_scores)

    summary = {
        "average_evidence_recall": avg_recall,
        "average_and_hop_success": avg_hop
    }

    all_results.append(summary)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nFINAL METRICS")
    print("Average Evidence Recall:", avg_recall)
    print("Average AND-Hop Success:", avg_hop)


