import os 
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

EVAL_V2_COARSE_CHUNK_COLLECTION_NAME = os.getenv("EVAL_V2_COARSE_CHUNK_COLLECTION_NAME")
EVAL_V2_SUMMARY_COLLECTION_NAME = os.getenv("EVAL_V2_SUMMARY_COLLECTION_NAME", "ev2_summary_muruga_v2")

NDIMS_LARGE_MODEL = os.getenv("EMBEDDING_DIMENSION_LARGE_MODEL", "1024")

LARGE_EMBEDDING_MODEL_NAME = os.getenv("LARGE_EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5") 

LARGE_EMBEDDING_MODEL_LOCAL_PATH = os.getenv("LARGE_EMBEDDING_MODEL_LOCAL_PATH", "C:/Users/srini/.cache/huggingface/hub/models--BAAI--bge-large-en-v1.5")

client = QdrantClient(
    url="http://localhost:6333",
    timeout=500, 
    prefer_grpc=True
)

create_summary_collection = True
create_coarse_chunk_collection = True
create_fine_chunk_collection = False

# print(SUMMARY_QDRANT_COLLECTION)
# print(COARSE_CHUNKS_QDRANT_COLLECTION)
# print(FINE_CHUNKS_QDRANT_COLLECTION)

def create_collections():
    CURRENT_EMBEDDING_DIMS = NDIMS_LARGE_MODEL
    CURRENT_EMBEDDING_MODEL = LARGE_EMBEDDING_MODEL_NAME

    # Create Qdrant Collections for summary COLLECTION
    if(create_summary_collection or client.collection_exists(EVAL_V2_SUMMARY_COLLECTION_NAME) == False):
        client.create_collection(
            collection_name = EVAL_V2_SUMMARY_COLLECTION_NAME,
            vectors_config=VectorParams(size = CURRENT_EMBEDDING_DIMS, distance=Distance.COSINE )
        )

        print(f"Created Collection: {EVAL_V2_SUMMARY_COLLECTION_NAME}")   

    # Create Qdrant Collections for COURSE CHUNK COLLECTION
    if(create_coarse_chunk_collection or client.collection_exists(EVAL_V2_COARSE_CHUNK_COLLECTION_NAME) == False):
        client.create_collection(
            collection_name = EVAL_V2_COARSE_CHUNK_COLLECTION_NAME,
            vectors_config=VectorParams(size = CURRENT_EMBEDDING_DIMS, distance=Distance.COSINE )
        )

        print(f"Created Collection: {EVAL_V2_COARSE_CHUNK_COLLECTION_NAME}")  

    # # Create Qdrant Collections for FINE CHUNK COLLECTION
    # if(create_fine_chunk_collection or client.collection_exists(FINE_CHUNKS_QDRANT_COLLECTION) == False):
    #     client.create_collection(
    #         collection_name = FINE_CHUNKS_QDRANT_COLLECTION,
    #         vectors_config=VectorParams(size = CURRENT_EMBEDDING_DIMS, distance=Distance.COSINE )
    #     )
    #     print(f"Created Collection: {FINE_CHUNKS_QDRANT_COLLECTION}")  

def get_qdrant_client():
    client = QdrantClient(
        url="http://localhost:6333",
        timeout=500, 
        prefer_grpc=True
    )
    return client

def get_summary_store():
    qdrant_client = get_qdrant_client()
    embedding_model = HuggingFaceEmbeddings(model_name=LARGE_EMBEDDING_MODEL_LOCAL_PATH)

    summary_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=EVAL_V2_SUMMARY_COLLECTION_NAME,
            embedding=embedding_model  
    )
    
    return summary_store 

def get_coarse_chunk_store():
    qdrant_client = get_qdrant_client()
    embedding_model = HuggingFaceEmbeddings(model_name=LARGE_EMBEDDING_MODEL_LOCAL_PATH)

    coarse_chunk_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=EVAL_V2_COARSE_CHUNK_COLLECTION_NAME,
            embedding=embedding_model  
     )
    return coarse_chunk_store


if(__name__ == "__main__"):
    create_collections() 
