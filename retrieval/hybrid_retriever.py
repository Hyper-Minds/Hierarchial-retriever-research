from collections import defaultdict
import os
from typing import List, Dict
from langchain_qdrant import QdrantVectorStore
from vectorstore.set_up_collections import get_coarse_chunk_store, get_summary_store
import traceback
from ingestion import metadata 
from dotenv import load_dotenv

load_dotenv()
print(1)

maximum_chunks_per_doc_to_return = int(os.getenv("MAXIMUM_CHUNKS_PER_DOC_TO_RETURN", 3))

def hybrid_document_retriever(
    query: str,
    summary_store : QdrantVectorStore,
    chunk_store : QdrantVectorStore,
    *,
    top_k_summaries: int = 0,
    top_k_chunks: int = 0,
    final_top_k_docs: int = 0,
    a: float = 0,
    b: float = 0
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
        summary_text : Dict[str, str] = {}
        summary_metadata = {}

        for doc, score in summary_results:
            doc_id = doc.metadata["summary_id"]
            summary_scores[doc_id] = score
            summary_text[doc_id] = doc.page_content 
            summary_metadata[doc_id] = doc.metadata


        # -----------------------------
        # 2. Query CHUNK collection
        # -----------------------------

        chunk_results = chunk_store.similarity_search_with_score(
            query,
            k=top_k_chunks
        )
        print("Retrieved Chunks: ", len(chunk_results))

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
            # after score add the chunk order to get the chunks in correct order later

            doc_id_to_chunks[doc_id].append(
                (
                    score, # add chunk order also 
                    {
                        "chunk_id": chunk_doc.metadata.get("chunk_id", "N/A"),
                        "chunk_content": chunk_doc.page_content
                    }
                )
            )
        
        print("[CHUNK STORE]: Processed Chunks per Document")

        # -----------------------------
        # 3. Combine scores per document
        # -----------------------------

        final_doc_scores = {}
    
        all_doc_ids = set(summary_scores.keys()) | set(chunk_scores.keys())
    
        for doc_id in all_doc_ids:
            s_score = summary_scores.get(doc_id, 0.0)
            c_score = chunk_scores.get(doc_id, 0.0)
            # print(f"Summary_score: {s_score} , Chunk Score: {c_score}")
    
            final_score = a * s_score + b * c_score
            final_doc_scores[doc_id] = final_score

        # -----------------------------
        # 4. Rank documents
        # -----------------------------
        ranked_docs = sorted(
            final_doc_scores.items(),
            key = lambda x: x[1],
            reverse = True
        )

        # -----------------------------
        # 5. Return top documents
        # -----------------------------
        results = []

        for doc_id, score in ranked_docs:
            chunks_list = doc_id_to_chunks.get(doc_id, [])
            
            # sort chunks based on score descending
            chunks_list = sorted(
                chunks_list,
                key=lambda x: x[0],
                reverse=True
            )

            # keep only the chunk dict, discard score and limit number of chunks
            final_limited_chunks = []

            for i, (chunk_score, chunk_dict) in enumerate(chunks_list):
                if i >= maximum_chunks_per_doc_to_return:
                    break
                final_limited_chunks.append(chunk_dict)

            final_supporting_chunks = len(doc_id_to_chunks.get(doc_id, [])) 

            document_medatata = metadata.get_metadata_from_summary_id(doc_id)

            results.append({
                "doc_id": doc_id,
                "final_score": score,
                "summary_text" : summary_text.get(doc_id, ""),
                "summary_score": summary_scores.get(doc_id, 0.0),
                "max_chunk_score": chunk_scores.get(doc_id, 0.0),
                "chunks": final_limited_chunks,
                "supporting_chunks" : final_supporting_chunks,
                "year" : document_medatata.get("year", ""),
                "cnr" : document_medatata.get("cnr", ""),
                "case_id" : document_medatata.get("case_id", ""),
                "respondent" : document_medatata.get("respondent", ""), 
                "petitioner" : document_medatata.get("petitioner", ""),
                "judge" : document_medatata.get("judge", ""),
                "court" : document_medatata.get("court", "")
            })

        return results[:final_top_k_docs]

    except Exception as e:
        print("ERROR DURING HYBRID RETRIEVAL:", str(e))
        traceback.print_exc()
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
        final_top_k_docs=2,
        a=0.5,
        b=0.5
    )

    # context = prompt.get_final_response_prompt(query, results)
    for doc in results:
        print(doc["doc_id"], doc["final_score"])
        print(len(doc["chunks"]))