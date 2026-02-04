def get_prompt(query, chunk_text, chunk_metadata, chunk_similarity_score):
    context = "\n\n"

    no_chunks = len(chunk_text)

    for i in range(no_chunks):
        chunk = f"""
        Chunk No: {i}
        Respondent: {chunk_metadata[i]["respondent"]}
        Petitioner: {chunk_metadata[i]["petitioner"]}

        Chunk Text : {chunk_text[i]}
        *chunk {i} end*
        """
        context = context + chunk 
        context = context + "\n\n"

    prompt = f"""
    You are a faithful Legal Assisstant, who helps users with making the legal research easier. 
    Answer the question strictly using ONLY the provided context.
    Do Not use any external knowledge.
    If you can not find the needed information, reply that information can not be found.
    Keep Your answer concise and point wise and less than 500 words. Use correct newlines for structuring the output

    User Query : {query}

    Context Chunks:
    {context}
    """

    return prompt


def get_final_response_prompt(query, hybrid_retriever_result):
    context = ""
    doc_count = 0
    chunk_count = 0 
    for doc_dict in hybrid_retriever_result:
        chunk_count = 0
        doc_context = f"""

            ---------- Case Judgement Document No: {doc_count} ---------
            Judgement Summary: 
            
            {doc_dict["summary_text"]}

            ---------- Document Chunks for Judgement document No {doc_count} are as follows : ----------

        """

        for chunk_dict in doc_dict["chunks"]:
            chunk_context = f"""
                ***** Chunk No: {chunk_count} for judgement document No: {doc_count} *****

                Chunk Text: 

                {chunk_dict["chunk_content"]}

                *** chunk {chunk_count} end ***
            """
            doc_context = doc_context + chunk_context + "\n\n"
            chunk_count += 1
        context = context + doc_context 
        doc_count += 1

    response_prompt = f"""
    ## SYSTEM ROLE:
    You are an expert legal knowledge engineer specializing in answering complex legal queries of users related to Case documents and court proceedings. 

    ## TASK:
    Using the provided document summaries and chunks, answer in detail to the user query related to the case documents and court proceedings.
    Use only the relevant and needed chunks and summaries to answer the user query. Ignore unrelated summaries and chunks. 
    Do not specify the document number or chunk number in your final answer.
    While specifying a judgemnt, mention the respondent and petetioner name.
    Do Not USE YOUR INTERNAL KNOWLEDGE to answer the user query. Use only the context to answer the user query. 
    If you can not find the needed information, reply that information can not be found.

    for example:
    The Supreme Court in the case of *Respondent Name* vs *Petetioner Name* held that...

    Give the answer in a concise and point wise manner. Use correct newlines for structuring the output. 
    The answer should be concise and less than 200 words.
    
    ## CONTEXT:
    {context}

    ## User Query:
    {query}

    """

    return response_prompt
    
def get_summary_prompt(metadata , document_text):
    document_metadata = f"""
        Title: {metadata["title"]}
        Court: {metadata["court"]}
        Respondent: {metadata["respondent"]}
        Petitioner: {metadata["petitioner"]}
        Judges: {metadata["judge"]} 
        Author Judge: {metadata["author_judge"]}
        CNR number: {metadata["cnr"]}
        Citation: {metadata["citation"]}
        disposal_nature : {metadata["disposal_nature"]}
        decision_date: {metadata["decision_date"]}
        Court: {metadata["court"]}
        Year : {metadata["year"]}
    """

    summary_prompt = f"""
    ## SYSTEM ROLE:
    You are an expert legal knowledge engineer specializing in extracting essential information from case judgements.

    ## TASK:
    Extract the below asked topics from the provided legal case jugement which is used for ingesting in a RAG pipeline.
    Each Topic must be given as: *topic_name* : "extracted_value". 

    DOCUMENT TEXT:
    {document_text}

    STRICT OUTPUT RULES (MANDATORY):
    - Do NOT write paragraphs
    - The output MUST contain ALL listed headers in the SAME order
    - No introductory or colclusion paragraphs
    - Do Not add any external information on your own. Only extract from the given document text.
    - If any topic information is NOT found in the document text, write "Not Available" as its value.

    TARGET LENGTH:
    ~250–300 words (dense, not verbose)

    TOPICS TO EXTRACT:
    ## Court Name:
    ## Petitioner Name(s):
    ## Respondent Name(s):
    ## Case Background: (write the factual background and problem which led to the case filing )
    ## Petitioner Arguments:
    ## Respondent Argument:
    ## Issues discussed in Court:
    ## Final Decision:
    ## Court's Reasoning:
    ## List of Important Precedents Cited:
    ## List of Acts:
    """

    return summary_prompt