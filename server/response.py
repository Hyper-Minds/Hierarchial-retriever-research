from llm import llama, prompt
from retrieval import hybrid_retriever
from vectorstore import set_up_collections


def get_response(query, summary_store, coarse_chunk_store):
    final_top_k_docs = 3
    top_k_summaries: int = 10
    top_k_chunks: int = 50
    final_top_k_docs: int = 5
    a = 0.7
    b = 0.3
    
    hybrid_retriever_result =  hybrid_retriever.hybrid_document_retriever(
        query,
        summary_store,
        coarse_chunk_store,
        top_k_summaries=top_k_summaries,
        top_k_chunks=top_k_chunks,
        final_top_k_docs=final_top_k_docs,
        a=a,
        b=b
    )

    print("[Response Generation]: Retrieval completed")
    
    print("DOCUMENT SUMMARIES RETRIEVED")
    for doc in hybrid_retriever_result:
        print(doc["doc_id"])
    
    final_response_prompt = prompt.get_final_response_prompt(query, hybrid_retriever_result)
    print("[Response Generation]: Prompt prepared, sending to LLM")
    
    llm_response = llama.get_answer(final_response_prompt)
    print("[Response Generation]: Response received from LLM")

    # print("----- FINAL RESPONSE FROM LLM -----")
    # print(llm_response)

    return llm_response, hybrid_retriever_result


if (__name__ == "__main__"):
    query = """ A complaint was filed by Sh. Neeraj Kumar on behalf of the appellant-firm, M/s Naresh Potteries. what was that. """
    summary_store = set_up_collections.get_summary_store()
    coarse_chunk_store = set_up_collections.get_coarse_chunk_store()
 
    res = get_response(query, summary_store, coarse_chunk_store)
    print(res)
    