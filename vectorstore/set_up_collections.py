import os 
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

SUMMARY_QDRANT_COLLECTION = os.getenv("SUMMARY_QDRANT_COLLECTION")
COARSE_CHUNKS_QDRANT_COLLECTION = os.getenv("COARSE_CHUNKS_QDRANT_COLLECTION")
FINE_CHUNKS_QDRANT_COLLECTION = os.getenv("FINE_CHUNKS_QDRANT_COLLECTION", "qdrant_fine_chunks_coll_new_V4")

NDIMS_LARGE_MODEL = os.getenv("EMBEDDING_DIMENSION_LARGE_MODEL", "1024")
NDIMS_SMALL_MODEL = os.getenv("EMBEDDING_DIMENSION_SMALL_MODEL", "768")

LARGE_EMBEDDING_MODEL_NAME = os.getenv("LARGE_EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5") 
SMALL_EMBEDDING_MODEL_NAME = os.getenv("SMALL_EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")  

LARGE_EMBEDDING_MODEL_LOCAL_PATH = os.getenv("LARGE_EMBEDDING_MODEL_LOCAL_PATH", "C:/Users/srini/.cache/huggingface/hub/models--BAAI--bge-large-en-v1.5")

client = QdrantClient(
    url="http://localhost:6333",
    timeout=500, 
    prefer_grpc=True
)

create_summary_collection = False 
create_coarse_chunk_collection = False
create_fine_chunk_collection = False

# print(SUMMARY_QDRANT_COLLECTION)
# print(COARSE_CHUNKS_QDRANT_COLLECTION)
# print(FINE_CHUNKS_QDRANT_COLLECTION)

def create_collections():
    CURRENT_EMBEDDING_DIMS = NDIMS_LARGE_MODEL
    CURRENT_EMBEDDING_MODEL = LARGE_EMBEDDING_MODEL_NAME

    # Create Qdrant Collections for summary COLLECTION
    if(create_summary_collection or client.collection_exists(SUMMARY_QDRANT_COLLECTION) == False):
        client.create_collection(
            collection_name = SUMMARY_QDRANT_COLLECTION,
            vectors_config=VectorParams(size = CURRENT_EMBEDDING_DIMS, distance=Distance.COSINE )
        )

        print(f"Created Collection: {SUMMARY_QDRANT_COLLECTION}")   

    # Create Qdrant Collections for COURSE CHUNK COLLECTION
    if(create_coarse_chunk_collection or client.collection_exists(COARSE_CHUNKS_QDRANT_COLLECTION) == False):
        client.create_collection(
            collection_name = COARSE_CHUNKS_QDRANT_COLLECTION,
            vectors_config=VectorParams(size = CURRENT_EMBEDDING_DIMS, distance=Distance.COSINE )
        )

        print(f"Created Collection: {COARSE_CHUNKS_QDRANT_COLLECTION}")  

    # Create Qdrant Collections for FINE CHUNK COLLECTION
    if(create_fine_chunk_collection or client.collection_exists(FINE_CHUNKS_QDRANT_COLLECTION) == False):
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
    embedding_model = HuggingFaceEmbeddings(model_name=LARGE_EMBEDDING_MODEL_LOCAL_PATH)

    summary_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=SUMMARY_QDRANT_COLLECTION,
            embedding=embedding_model  
    )
    
    return summary_store 

def get_coarse_chunk_store():
    qdrant_client = get_qdrant_client()
    embedding_model = HuggingFaceEmbeddings(model_name=LARGE_EMBEDDING_MODEL_LOCAL_PATH)

    coarse_chunk_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COARSE_CHUNKS_QDRANT_COLLECTION,
            embedding=embedding_model  
     )
    return coarse_chunk_store

def create_payload_index():
    create_index_for_summaries = True
    create_index_for_coarse_chunks = True

    if(create_index_for_summaries):

        # Create the Paylod for the field - cnr, case_id, respondent, petitioner, judge

        # Create Payload Index on 'cnr' field for SUMMARY COLLECTION
        client.create_payload_index(
            collection_name=SUMMARY_QDRANT_COLLECTION,
            field_name="cnr",
            field_schema = models.TextIndexParams(
                type = models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase = True,
                phrase_matching = True
            )            
        )
        print(f"Created Payload Index on 'cnr' for Collection: {SUMMARY_QDRANT_COLLECTION}")

        # Create Payload Index for "Case ID"
        client.create_payload_index(
            collection_name=SUMMARY_QDRANT_COLLECTION,
            field_name="case_id",
            field_schema = models.TextIndexParams(
                type = models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase = True,
                phrase_matching = True
            )            
        )
        
        print(f"Created Payload Index on 'case_id' for Collection: {SUMMARY_QDRANT_COLLECTION}")

        # Create Payload Index for "petitioner"
        client.create_payload_index(
            collection_name=SUMMARY_QDRANT_COLLECTION,
            field_name="petitioner",
            field_schema = models.TextIndexParams(
                type = models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase = True,
                phrase_matching = True
            )            
        )
        print(f"Created Payload Index on 'petitioner' for Collection: {SUMMARY_QDRANT_COLLECTION}")

        # Create Payload Index for "respondent"
        client.create_payload_index(
            collection_name=SUMMARY_QDRANT_COLLECTION,
            field_name="respondent",
            field_schema = models.TextIndexParams(
                type = models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase = True,
                phrase_matching = True
            )            
        )
        print(f"Created Payload Index on 'respondent' for Collection: {SUMMARY_QDRANT_COLLECTION}")

        # Create Payload for "judge"
        client.create_payload_index(
            collection_name=SUMMARY_QDRANT_COLLECTION,
            field_name="judge",
            field_schema = models.TextIndexParams(  
                type = models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase = True,
                phrase_matching = True
            )
        )            
        print(f"Created Payload Index on 'judge' for Collection: {SUMMARY_QDRANT_COLLECTION}")

    if(create_index_for_coarse_chunks):
        # Create the Paylod for the field - cnr, case_id, respondent, petitioner, judge

        # Create Payload Index on 'cnr' field for COARSE CHUNKS COLLECTION
        client.create_payload_index(
            collection_name=COARSE_CHUNKS_QDRANT_COLLECTION,
            field_name="cnr",
            field_schema = models.TextIndexParams(
                type = models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase = True,
                phrase_matching = True
            )            
        )
        print(f"Created Payload Index on 'cnr' for Collection: {COARSE_CHUNKS_QDRANT_COLLECTION}")

        # Create Payload Index for "Case ID"
        client.create_payload_index(
            collection_name=COARSE_CHUNKS_QDRANT_COLLECTION,
            field_name="case_id",
            field_schema = models.TextIndexParams(
                type = models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase = True,
                phrase_matching = True
            )            
        )

        print(f"Created Payload Index on 'case_id' for Collection: {COARSE_CHUNKS_QDRANT_COLLECTION}")

        # Create Payload Index for "petitioner"
        client.create_payload_index(
            collection_name=COARSE_CHUNKS_QDRANT_COLLECTION,
            field_name="petitioner",
            field_schema = models.TextIndexParams(
                type = models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase = True,
                phrase_matching = True
            )            
        )
        print(f"Created Payload Index on 'petitioner' for Collection: {COARSE_CHUNKS_QDRANT_COLLECTION}")

        # Create Payload Index for "respondent"
        client.create_payload_index(
            collection_name=COARSE_CHUNKS_QDRANT_COLLECTION,
            field_name="respondent",
            field_schema = models.TextIndexParams(
                type = models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase = True,
                phrase_matching = True
            )            
        )
        print(f"Created Payload Index on 'respondent' for Collection: {COARSE_CHUNKS_QDRANT_COLLECTION}")

        # Create Payload for "judge"
        client.create_payload_index(
            collection_name=COARSE_CHUNKS_QDRANT_COLLECTION,
            field_name="judge",
            field_schema = models.TextIndexParams(  
                type = models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase = True,
                phrase_matching = True
            )
        )            
        print(f"Created Payload Index on 'judge' for Collection: {COARSE_CHUNKS_QDRANT_COLLECTION}")



if(__name__ == "__main__"):
    create_collections() 
    create_payload_index()
