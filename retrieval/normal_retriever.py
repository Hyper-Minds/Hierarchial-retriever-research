from collections import defaultdict
from typing import List, Dict
from langchain_qdrant import QdrantVectorStore
from vectorstore.set_up_collections import get_coarse_chunk_store
import os

NORMAL_RETRIEVER_EVALUATION_OUTPUT_FILE = os.getenv("NORMAL_RETRIEVER_EVALUATION_OUTPUT_FILE", "normal_retriever_results.json")

def normal_retriever(
    query: str,
    chunk_store : QdrantVectorStore,
    top_k_chunks: int = 50,
    final_top_k_docs: int = 10
):
    """
    Hybrid document-level retriever using:
    final_score = a * summary_score + b * max_chunk_score
    """
    try:
        chunk_results = chunk_store.similarity_search_with_score(
            query,
            k=top_k_chunks
        )

        # doc_id -> max_chunk_score
        chunk_scores: Dict[str, List] = defaultdict(list)
        doc_id_to_chunks: Dict[str, List[Dict]] = defaultdict(list)

        print("[COARSE CHUNK STORE]: Retrieved Coarse Chunks")

        for chunk_doc, score in chunk_results:
            doc_id = chunk_doc.metadata["doc_id"]

            # # keep only the strongest chunk per document
            # if score > chunk_scores[doc_id]:
            chunk_scores[doc_id].append(score)

            # Store the chunks under the corresponding doc id
            doc_id_to_chunks[doc_id].append({
                "doc_id": doc_id,
                "chunk_id": chunk_doc.metadata.get("chunk_id", "N/A"),
                # "chunk_text": chunk_doc.page_content,
                "score": score
            })

        final_doc_scores = {}

        all_doc_ids = set(chunk_scores.keys())

        for doc_id in all_doc_ids:
            num_chunks = len(chunk_scores[doc_id])
            scores_sum = sum(chunk_scores[doc_id])
            print("Sum of chunk scores: ", scores_sum, " No of chunks: ", num_chunks)

            avg_score = scores_sum / num_chunks
            final_doc_scores[doc_id] = avg_score

        # -----------------------------
        # 4. Rank documents
        # -----------------------------
        ranked_docs = sorted(
            final_doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # -----------------------------
        # 5. Return top documents
        # -----------------------------
        results = []

        for doc_id, score in ranked_docs[:final_top_k_docs]:
            results.append({
                "doc_id": doc_id,
                "chunk_score": score,
                "maximum_rel_chunk_id" : [],
                "chunks": doc_id_to_chunks.get(doc_id, [])
            })

        return results
    
    except Exception as e:
        print("ERROR DURING HYBRID RETRIEVAL:", str(e))
        return []