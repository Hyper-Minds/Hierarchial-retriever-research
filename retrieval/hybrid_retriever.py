from collections import defaultdict
from typing import List, Dict
from langchain_qdrant import QdrantVectorStore
from vectorstore.set_up_collections import get_coarse_chunk_store, get_summary_store

def hybrid_document_retriever(
    query: str,
    summary_store : QdrantVectorStore,
    chunk_store : QdrantVectorStore,
    *,
    top_k_summaries: int = 10,
    top_k_chunks: int = 50,
    final_top_k_docs: int = 5,
    a: float = 0.7,
    b: float = 0.3
):
    """
    Hybrid document-level retriever using:
    final_score = a * summary_score + b * max_chunk_score
    """

    # assert a > b, "Design constraint violated: a must be > b"

    # -----------------------------
    # 1. Query SUMMARY collection
    # -----------------------------

    try:
        summary_results = summary_store.similarity_search_with_score(
            query,
            k=top_k_summaries
        )
        # summary_id -> summary_score
        print("[SUMMARY STORE]: Retrieved Summary")
        summary_scores: Dict[str, float] = {}

        for doc, score in summary_results:
            doc_id = doc.metadata["summary_id"]
            summary_scores[doc_id] = score

        # -----------------------------
        # 2. Query CHUNK collection
        # -----------------------------
        chunk_results = chunk_store.similarity_search_with_score(
            query,
            k=top_k_chunks
        )

        # doc_id -> max_chunk_score
        chunk_scores: Dict[str, float] = defaultdict(float)
        doc_id_to_chunks: Dict[str, List[Dict]] = defaultdict(list)
        print("[COARSE RETRIEVER]: Retrieved Coarse Chunks")

        for chunk_doc, score in chunk_results:
            doc_id = chunk_doc.metadata["doc_id"]

            # keep only the strongest chunk per document
            if score > chunk_scores[doc_id]:
                chunk_scores[doc_id] = score

            # Store the chunks under the corresponding doc id
            doc_id_to_chunks[doc_id].append({
                "doc_id": doc_id,
                "chunk_id": chunk_doc.metadata.get("chunk_id", "N/A"),
                "chunk_text": chunk_doc.page_content,
                "score": score
            })

            # print(len(chunk_doc.page_content))

        # -----------------------------
        # 3. Combine scores per document
        # -----------------------------
        final_doc_scores = {}

        all_doc_ids = set(summary_scores.keys()) | set(chunk_scores.keys())

        for doc_id in all_doc_ids:
            s_score = summary_scores.get(doc_id, 0.0)
            c_score = chunk_scores.get(doc_id, 0.0)

            final_score = a * s_score + b * c_score
            final_doc_scores[doc_id] = final_score

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
                "final_score": score,
                "summary_score": summary_scores.get(doc_id, 0.0),
                "chunk_score": chunk_scores.get(doc_id, 0.0),
                "chunks": doc_id_to_chunks.get(doc_id, [])
            })

        return results
    
    except Exception as e:
        print("ERROR DURING HYBRID RETRIEVAL:", str(e))
        return []

if(__name__ == "__main__"):

    query = "M/s Naresh Potteries and M/s Aarti Industries and Another. Name the judge of the case"
    summary_store = get_summary_store()
    chunk_store = get_coarse_chunk_store()
    results = hybrid_document_retriever(
        query,
        summary_store,
        chunk_store,
        top_k_summaries=10,
        top_k_chunks=50,
        final_top_k_docs=10,
        a=0.7,
        b=0.3
    )
    for (i, res) in enumerate(results):
        print("_" * 10 + f"Result {i+1}:" + "_" * 10)
        print(res)