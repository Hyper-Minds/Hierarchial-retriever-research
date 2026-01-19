from langchain_huggingface import HuggingFaceEmbeddings
import os 

LARGE_EMBEDDING_MODEL_NAME = os.getenv("LARGE_EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")  
SMALL_EMBEDDING_MODEL_NAME = os.getenv("SMALL_EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")

def get_large_embeddingModel():
    embedding_model = HuggingFaceEmbeddings(model_name=LARGE_EMBEDDING_MODEL_NAME)
    return embedding_model

def get_small_embeddingModel():
    embedding_model = HuggingFaceEmbeddings(model_name=SMALL_EMBEDDING_MODEL_NAME)
    return embedding_model

if(__name__ == "__main__"):
    large_model = get_large_embeddingModel()
    small_model = get_small_embeddingModel()
    print("Large and Small Embedding Models Loaded Successfully.")