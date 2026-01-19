import os 
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

SUMMARY_QDRANT_COLLECTION = os.getenv("SUMMARY_QDRANT_COLLECTION")
COARSE_CHUNKS_QDRANT_COLLECTION = os.getenv("COARSE_CHUNKS_QDRANT_COLLECTION")
FINE_CHUNKS_QDRANT_COLLECTION = os.getenv("FINE_CHUNKS_QDRANT_COLLECTION", "qdrant_fine_chunks_coll_new_V4")

NDIMS_LARGE_MODEL = os.getenv("EMBEDDING_DIMENSION_LARGE_MODEL", "1024")
NDIMS_SMALL_MODEL = os.getenv("EMBEDDING_DIMENSION_SMALL_MODEL", "768")

LARGE_EMBEDDING_MODEL_NAME = os.getenv("LARGE_EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5") 
SMALL_EMBEDDING_MODEL_NAME = os.getenv("SMALL_EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")  

client = QdrantClient(
    url="http://localhost:6333",
    timeout=120, 
    prefer_grpc=True
)

# print(SUMMARY_QDRANT_COLLECTION)
# print(COARSE_CHUNKS_QDRANT_COLLECTION)
# print(FINE_CHUNKS_QDRANT_COLLECTION)

def create_collections():
    CURRENT_EMBEDDING_DIMS = NDIMS_LARGE_MODEL
    CURRENT_EMBEDDING_MODEL = LARGE_EMBEDDING_MODEL_NAME

    # Create Qdrant Collections for summary COLLECTION
    if(client.collection_exists(SUMMARY_QDRANT_COLLECTION) == False):
        client.delete_collection(SUMMARY_QDRANT_COLLECTION)

    client.create_collection(
        collection_name = SUMMARY_QDRANT_COLLECTION,
        vectors_config=VectorParams(size = CURRENT_EMBEDDING_DIMS, distance=Distance.COSINE )
    )

    print(f"Created Collection: {SUMMARY_QDRANT_COLLECTION}")   

    # Create Qdrant Collections for COURSE CHUNK COLLECTION
    if(client.collection_exists(COARSE_CHUNKS_QDRANT_COLLECTION) == False):
        client.delete_collection(COARSE_CHUNKS_QDRANT_COLLECTION)

    client.create_collection(
        collection_name = COARSE_CHUNKS_QDRANT_COLLECTION,
        vectors_config=VectorParams(size = CURRENT_EMBEDDING_DIMS, distance=Distance.COSINE )
    )

    print(f"Created Collection: {COARSE_CHUNKS_QDRANT_COLLECTION}")  

    # Create Qdrant Collections for FINE CHUNK COLLECTION
    if(client.collection_exists(FINE_CHUNKS_QDRANT_COLLECTION) == False):
        client.delete_collection(FINE_CHUNKS_QDRANT_COLLECTION)

    client.create_collection(
        collection_name = FINE_CHUNKS_QDRANT_COLLECTION,
        vectors_config=VectorParams(size = CURRENT_EMBEDDING_DIMS, distance=Distance.COSINE )
    )
    print(f"Created Collection: {FINE_CHUNKS_QDRANT_COLLECTION}")  

def get_qdrant_client():
    client = QdrantClient(
        url="http://localhost:6333",
        timeout=500, 
        prefer_grpc=True
    )
    return client

def get_summary_store():
    qdrant_client = get_qdrant_client()
    embedding_model = HuggingFaceEmbeddings(model_name=LARGE_EMBEDDING_MODEL_NAME)

    summary_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=SUMMARY_QDRANT_COLLECTION,
            embedding=embedding_model  
        )
    
    return summary_store 

def get_coarse_chunk_store():
    qdrant_client = get_qdrant_client()
    embedding_model = HuggingFaceEmbeddings(model_name=LARGE_EMBEDDING_MODEL_NAME)

    coarse_chunk_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COARSE_CHUNKS_QDRANT_COLLECTION,
            embedding=embedding_model  
        )
    return coarse_chunk_store


if(__name__ == "__main__"):
    create_collections() 