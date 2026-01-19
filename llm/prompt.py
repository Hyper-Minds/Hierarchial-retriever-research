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
