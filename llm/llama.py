import ollama

def get_answer(prompt: str):
    """
    Runs Llama 3.2 with specific quantization.
    Default '3b' is 4-bit.
    """
    model_name = f"llama3.2:3b"
    
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={
            "num_gpu": 28,      # Force layers to GPU (Llama 3.2 3B has 28 layers)
            "num_ctx": 8192,    # Crucial for 4GB GPU: limits memory used by 'context'
            "temperature": 0.2  # Lower temperature is better for RAG/Reasoning
        }
    )
    return response['response']


# if(__name__ == "__main__"):
#     # To use the default 4-bit (recommended):
#     query = """M/s Naresh Potteries and M/s Aarti Industries and Another. Name the judge of the case"""
#     embedding_model = embeddings.get_hugging_face_embeddingModel()

#     chunks_retrieved, chunks_metadata, chunks_score = retriever.retrieve(query, embedding_model, top_k = 2)

#     context = f"""
#     Chunk Text 1: 
#     {chunks_retrieved[0]}

#     Metadata : {chunks_metadata[0]}    

#     Chunk Text 2:
#     {chunks_retrieved[1]}

#     Chunk Metadata 2:
#     {chunks_metadata[1]}
#     """

#     prompt = f""""
#     You are a faithful Leagl Assisstant, who helps users with making the legal research easier. 
#     You reply to the user query based on the context provided to you. Keep the response less than 500 words.

#     User Query : {query}

#     Context Chunks:
#     {context}

#     Keep Your answer concise and point wise.
#     """

#     print(get_answer(prompt))

#     # To use 8-bit (if you want more reasoning power and have VRAM to spare):