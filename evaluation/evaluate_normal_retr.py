import os 
import json
from dotenv import load_dotenv
from retrieval import normal_retriever
from evaluation import metrics
from vectorstore.set_up_collections import get_coarse_chunk_store, get_summary_store

load_dotenv()

EVALUATION_RESULTS_FOLDER = os.getenv("EVALUATION_RESULTS_FOLDER", "C:/Tejeswar/AI Research Engine/Final_Project/evaluation/results/")
EVALUATION_GROUND_TRUTH_FILE = os.getenv("EVALUATION_GROUND_TRUTH_FILE", "ground_truth.json")
NORMAL_RETRIEVER_EVALUATION_OUTPUT_FILE = os.getenv("NORMAL_RETRIEVER_EVALUATION_OUTPUT_FILE", "normal_retriever_results.json")
ground_truth_doc_id = []


def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["queries"]
    

if(__name__ == "__main__"):
    ground_truth_path = os.path.join(EVALUATION_RESULTS_FOLDER, EVALUATION_GROUND_TRUTH_FILE)
    ground_truth_data = load_ground_truth(ground_truth_path)

    K = 10
    top_k_chunks = 10

    evaluation_precision = []
    evaluation_recall = []
    evaluation_result = []

    global_retrieved_lst = [] 
    global_relevent_lst = [] 

    coarse_chunk_vector_store = get_coarse_chunk_store()

    for query_data in ground_truth_data:
        query_text = query_data["query"]
        relevant_docs = query_data["relevant_docs"]

        retrieved_docs = normal_retriever.normal_retriever(
            query= query_text,
            chunk_store = coarse_chunk_vector_store,
            top_k_chunks = top_k_chunks,
            final_top_k_docs = K,
        )
        retrieved_doc_ids = [doc["doc_id"] for doc in retrieved_docs]

        print("[EVALUATE]  Retrieved Docs for Query:", query_text)

        precision, recall = metrics.precision_recall_at_k(retrieved_doc_ids, relevant_docs, k = K)
        
        evaluation_precision.append(precision)
        evaluation_recall.append(recall)

        
        chunks_score = [doc["chunk_score"] for doc in retrieved_docs]
        max_rel_chunk_id_lst = [doc.get("maximum_rel_chunk_id", []) for doc in retrieved_docs]

        # append the documents to the global retrieved and global relevent doc id list 
        global_relevent_lst.append(relevant_docs)
        global_retrieved_lst.append(retrieved_doc_ids)

        evaluation_result.append({
            "query": query_text,
            "relevant_docs_id": relevant_docs,
            "retrieved_docs_id": retrieved_doc_ids,
            "chunks_score": chunks_score,
            "maximum_rel_chunk_ids" : max_rel_chunk_id_lst,
            f"precision_at{K}": precision,
            f"recall_at@{K}": recall
        })

        print(f"Precision@{K}: {precision:.4f}, Recall@{K}: {recall:.4f}")
        print("")

    avg_precision = sum(evaluation_precision) / len(evaluation_precision)
    avg_recall = sum(evaluation_recall) / len(evaluation_recall)
    mean_reciprocal_rank = metrics.get_mean_reciprocal_rank(global_retrieved_lst, global_relevent_lst)
    mean_norm_dcg_score = metrics.mean_ndcg_at_k(global_retrieved_lst, global_relevent_lst, K)

    print()
    print(f"Average Precision@{K}: {avg_precision:.4f}")
    print(f"Average Recall@{K}: {avg_recall:.4f}")
    print(f"Mean Reciprocal Rank: {mean_reciprocal_rank:.4f}")
    print(f"Mean nDCG@{K} : {mean_norm_dcg_score:.4f}")
    print()

    evaluation_result.append({
        f"average_precision_at_{K}": avg_precision,
        f"average_recall_at_{K}": avg_recall   ,
        f"mean_reciprocal_rank" : mean_reciprocal_rank,
        f"mean_normalized_dcg@{K}" : mean_norm_dcg_score
    })

    output_path = os.path.join(EVALUATION_RESULTS_FOLDER, NORMAL_RETRIEVER_EVALUATION_OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": evaluation_result}, f, ensure_ascii=False, indent=2)