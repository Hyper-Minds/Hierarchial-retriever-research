import os 
import json
from dotenv import load_dotenv
from retrieval import hybrid_retriever
from evaluation import metrics
from vectorstore.set_up_collections import get_coarse_chunk_store, get_summary_store

load_dotenv()

EVALUATION_RESULTS_FOLDER = os.getenv("EVALUATION_RESULTS_FOLDER", "C:/Tejeswar/AI Research Engine/Final_Project/evaluation/results/")
EVALUATION_GROUND_TRUTH_FILE = os.getenv("EVALUATION_GROUND_TRUTH_FILE", "ground_truth.json")
EVALUATION_OUTPUT_FILE = os.getenv("EVALUATION_OUTPUT_FILE", "retriever_results.json")


def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["queries"]
    

if(__name__ == "__main__"):
    ground_truth_path = os.path.join(EVALUATION_RESULTS_FOLDER, EVALUATION_GROUND_TRUTH_FILE)
    ground_truth_data = load_ground_truth(ground_truth_path)


    # define the parameters for retrieval
    top_k_summaries = 10
    top_k_chunks = 10
    final_top_k_docs = 10
    k = final_top_k_docs

    a_param = 0.5
    b_param = 0.5

    global_retrieved_lst = [] 
    global_relevent_lst = [] 

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
            top_k_summaries = top_k_summaries,
            top_k_chunks = top_k_chunks,
            final_top_k_docs = k,
            a = a_param,
            b = b_param
        )
        retrieved_doc_ids = [doc["doc_id"] for doc in retrieved_docs]

        print("[EVALUATE]  Retrieved Docs for Query:", query_text)

        precision, recall = metrics.precision_recall_at_k(retrieved_doc_ids, relevant_docs, k=k)
        
        evaluation_precision.append(precision)
        evaluation_recall.append(recall)

        
        # append the documents to the global retrieved and global relevent doc id list 
        global_relevent_lst.append(relevant_docs)
        global_retrieved_lst.append(retrieved_doc_ids)

        evaluation_result.append({
            "query": query_text,
            "relevant_docs_id": relevant_docs,
            "retrieved_docs_id": retrieved_doc_ids,
            f"precision_at_{k}": precision,
            f"recall_at_{k}": recall
        })

        print(f"Precision@{k}: {precision:.4f}, Recall@{k}: {recall:.4f}")
        print("")

    # print(global_retrieved_lst)

    print()

    # print(global_relevent_lst)
 
    avg_precision = sum(evaluation_precision) / len(evaluation_precision)
    avg_recall = sum(evaluation_recall) / len(evaluation_recall)
    mean_reciprocal_rank = metrics.get_mean_reciprocal_rank(global_retrieved_lst, global_relevent_lst)
    normalized_dcg = metrics.mean_ndcg_at_k(global_retrieved_lst, global_relevent_lst, k)
    
    
    print(f"Average Precision@{k}: {avg_precision:.4f}")
    print(f"Average Recall@{k}: {avg_recall:.4f}")
    print(f"Mean Reciprocal Rank: {mean_reciprocal_rank:.4f}")
    print(f"Mean nDCG@{k} : {normalized_dcg:.4f}")
    print()
 
    evaluation_result.append({
        f"average_precision_at_{k}": avg_precision,
        f"average_recall_at_{k}": avg_recall,
        "mean_reciprocal_rank" : mean_reciprocal_rank,
        f"mean_normalized_dcg@{k}" : normalized_dcg
    })
 
    output_path = os.path.join(EVALUATION_RESULTS_FOLDER, EVALUATION_OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": evaluation_result}, f, ensure_ascii=False, indent=2)
