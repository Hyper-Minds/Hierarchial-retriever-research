
import math

def precision_recall_at_k(retrieved_doc_id, relevant_doc_id, k):
    retrieved_doc_id = retrieved_doc_id[:k]

    retrieved_docs_set = set(retrieved_doc_id)
    relevant_docs_set = set(relevant_doc_id)

    true_positives = len(retrieved_docs_set & relevant_docs_set)

    precision = true_positives / k
    recall = true_positives / len(relevant_docs_set)

    return precision, recall

def get_mean_reciprocal_rank(retrieved_lst, relevent_lst):
    """
    retrieved_lst list of lists
        Each inner list is the ranked retrieved document IDs for one query.
    relevent_lst: list of sets
        Each set contains the relevant (ground-truth) document IDs for one query.
    Returns:
        Mean Reciprocal Rank (MRR) using binary relevance.
    """
     
    assert len(retrieved_lst) == len(relevent_lst)

    reciprocal_ranks = []
    q = 1
    for retrieved_doc_id, relevent_doc_id in zip(retrieved_lst, relevent_lst):
        # print(len(retrieved_doc_id), len(relevent_doc_id))
        rr = 0.0
        for rank, doc_id in enumerate(retrieved_doc_id, start=1):
            if doc_id in relevent_doc_id:
                rr = 1.0 / rank
                reciprocal_ranks.append(rr)
                print("query: ", q, "mean rr: ", rr)
                break
        else:
            reciprocal_ranks.append(0)
            print("No relevant docs for query: ", q, "mean rr: ", 0)

        q += 1

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def dcg_at_k(retrieved_doc_ids, relevant_doc_ids, k):
    """
    Computes DCG@k for a single query (binary relevance)
    If the retrieved_doc_id length is less than k, it computes dcg for that number of documents alone. 
    """

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        if doc_id in relevant_doc_ids:
            dcg += 1 / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def idcg_at_k(num_relevant_docs, k):
    """
    Computes IDCG@k for a single query (binary relevance)
    """
    ideal_hits = min(num_relevant_docs, k)
    idcg = 0.0
    for i in range(ideal_hits):
        idcg += 1 / math.log2(i + 2)
    return idcg


def ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k):
    """
    Computes nDCG@k for a single query
    """
    if len(relevant_doc_ids) == 0:
        return 0.0

    dcg = dcg_at_k(retrieved_doc_ids, relevant_doc_ids, k)
    idcg = idcg_at_k(len(relevant_doc_ids), k)

    return dcg / idcg if idcg > 0 else 0.0


def mean_ndcg_at_k(retrieved_lst, relevent_lst, k):
    """
    Computes mean nDCG@k over n queries
    """
    ndcg_scores = []  

    assert(len(retrieved_lst) == len(relevent_lst))
    q = 1

    for retrieved_docs, relevant_docs in zip(retrieved_lst, relevent_lst):
        score = ndcg_at_k(retrieved_docs, relevant_docs, k)
        print(f"nDCG for query {q} : {score}")
        ndcg_scores.append(score)
        q += 1

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
