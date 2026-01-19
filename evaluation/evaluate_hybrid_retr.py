import pandas as pd 
import os 
import json
from dotenv import load_dotenv
from retrieval import hybrid_retriever
from vectorstore.set_up_collections import get_coarse_chunk_store, get_summary_store

load_dotenv()

EVALUATION_RESULTS_FOLDER = os.getenv("EVALUATION_RESULTS_FOLDER", "C:/Tejeswar/AI Research Engine/Final_Project/evaluation/results/")
EVALUATION_GROUND_TRUTH_FILE = os.getenv("EVALUATION_GROUND_TRUTH_FILE", "ground_truth.json")
EVALUATION_OUTPUT_FILE = os.getenv("EVALUATION_OUTPUT_FILE", "retriever_results.json")

ground_truth_doc_id = []

def precision_recall_at_k(retrieved, relevant_docs, k):
    retrieved_k = retrieved[:k]
    retrieved_doc_ids = {r["doc_id"] for r in retrieved_k}
    relevant_docs = set(relevant_docs)

    true_positives = len(retrieved_doc_ids & relevant_docs)

    precision = true_positives / k
    recall = true_positives / len(relevant_docs)

    return precision, recall

def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["queries"]
    

if(__name__ == "__main__"):
    ground_truth_path = os.path.join(EVALUATION_RESULTS_FOLDER, EVALUATION_GROUND_TRUTH_FILE)
    ground_truth_data = load_ground_truth(ground_truth_path)

    k = 10

    evaluation_precision = []
    evaluation_recall = []
    evaluation_result = []

    summary_vector_store = get_summary_store()
    coarse_chunk_vector_store = get_coarse_chunk_store()

    for query_data in ground_truth_data:
        query_text = query_data["query"]
        relevant_docs = query_data["relevant_docs"]

        retrieved_docs = hybrid_retriever.hybrid_document_retriever(
            query = query_text,
            summary_store = summary_vector_store,
            chunk_store = coarse_chunk_vector_store,
            top_k_summaries = 10,
            top_k_chunks = 50,
            final_top_k_docs = k,
            a = 0.5,
            b = 0.5
        )

        print("[EVALUATE]  Retrieved Docs for Query:", query_text)

        precision, recall = precision_recall_at_k(retrieved_docs, relevant_docs, k=k)
        
        evaluation_precision.append(precision)
        evaluation_recall.append(recall)

        retrieved_doc_ids = [doc["doc_id"] for doc in retrieved_docs]

        evaluation_result.append({
            "query": query_text,
            "relevant_docs_id": relevant_docs,
            "retrieved_docs_id": retrieved_doc_ids,
            f"precision_at_{k}": precision,
            f"recall_at_{k}": recall
        })

        print(f"Precision@{k}: {precision:.4f}, Recall@{k}: {recall:.4f}")
        print("")

    avg_precision = sum(evaluation_precision) / len(evaluation_precision)
    avg_recall = sum(evaluation_recall) / len(evaluation_recall)
    print(f"Average Precision@{k}: {avg_precision:.4f}")
    print(f"Average Recall@{k}: {avg_recall:.4f}")
    print()

    evaluation_result.append({
        f"average_precision_at_{k}": avg_precision,
        f"average_recall_at_{k}": avg_recall   
    })


    output_path = os.path.join(EVALUATION_RESULTS_FOLDER, EVALUATION_OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": evaluation_result}, f, ensure_ascii=False, indent=2)

    # k_values = [1, 3, 5, 10]

    # overall_metrics = {k: {"precision": 0, "recall": 0} for k in k_values}
    # total_queries = len(ground_truth_data)

    # for query in ground_truth_data:
    #     relevant_docs = query["relevant_docs"]
    #     retrieved_docs = query["retrieved_docs"]

    #     for k in k_values:
    #         precision, recall = precision_recall_at_k(retrieved_docs, relevant_docs, k)
    #         overall_metrics[k]["precision"] += precision
    #         overall_metrics[k]["recall"] += recall

    # for k in k_values:
    #     overall_metrics[k]["precision"] /= total_queries
    #     overall_metrics[k]["recall"] /= total_queries
    #     print(f"At K={k}: Precision={overall_metrics[k]['precision']:.4f}, Recall={overall_metrics[k]['recall']:.4f}")