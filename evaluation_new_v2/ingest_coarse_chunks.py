import os
from langchain_qdrant import QdrantVectorStore
from uuid6 import uuid7
import json
from evaluation_new_v2.create_qdrant_collections import get_qdrant_client
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

print(1)

EVAL_V2_COARSE_CHUNK_COLLECTION_NAME = os.getenv("EVAL_V2_COARSE_CHUNK_COLLECTION_NAME")
EVAL_V2_SUMMARY_COLLECTION_NAME = os.getenv("EVAL_V2_SUMMARY_COLLECTION_NAME")

EVAL_DIR_V2 = os.getenv("EVAL_DIR_V2")
EVAL_V2_COARSE_CHUNKS_DIRECTORY= os.getenv("EVAL_V2_COARSE_CHUNKS_DIRECTORY")

EVAL_V2_RELEVENT_JUDGEMENT_LIST_JSON = os.getenv("EVAL_V2_RELEVENT_JUDGEMENT_LIST_JSON")
LARGE_EMBEDDING_MODEL_NAME = os.getenv("LARGE_EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")

def ingest_coarse_chunks():
    try:
        qdrant_client = get_qdrant_client()
        embedding_model = HuggingFaceEmbeddings(model_name=LARGE_EMBEDDING_MODEL_NAME)

        vector_store = QdrantVectorStore(
                client = qdrant_client,
                collection_name = EVAL_V2_COARSE_CHUNK_COLLECTION_NAME,
                embedding = embedding_model
            )
        
        document_id_list = []
        with open( os.path.join(EVAL_DIR_V2, EVAL_V2_RELEVENT_JUDGEMENT_LIST_JSON) ) as file:
            judgements_dict = json.load(file)
            document_id_list = judgements_dict["relevent_docs_id"]

        file_count = 0 

        for document_id in document_id_list:
            coarse_folder_path = os.path.join(EVAL_V2_COARSE_CHUNKS_DIRECTORY, document_id)
            chunk_files = [file_name for file_name in os.listdir(coarse_folder_path) if file_name.endswith(".json")] 

            # Stores chunks for a single summary document
            list_of_chunk_documents = []

            for chunk_file in chunk_files:
                # process each chunk file
                with open(os.path.join(coarse_folder_path, chunk_file), "r", encoding="utf-8") as f:
                    chunk_data = json.load(f)

                    # chunk data is the json for each chunk 
                    if("chunk_id" not in chunk_data):
                        chunk_data["chunk_id"] =  uuid7()
                        print("NEW CHUNK ID ADDED:", chunk_data["chunk_id"]) 

                    # CHECK IF CHUNK IS VALID FOR THIS SUMMARY
                    if("text" not in chunk_data or "doc_id" not in chunk_data):
                        print("INVALID CHUNK DATA, SKIPPING:", chunk_file)
                        continue

                    chunk_text = chunk_data["text"]
                    chunk_id = chunk_data["chunk_id"]
                    doc_id = chunk_data["doc_id"]
                    
                    chunk_doc_obj = Document(
                        page_content=chunk_text,
                    )

                    list_of_chunk_documents.append(chunk_doc_obj)

            vector_store.add_documents(list_of_chunk_documents)

            file_count += 1
            print(f"[INGESTION - COARSE CHUNKS]: {file_count} Document ID: {document_id} TOTAL CHUNKS: {len(list_of_chunk_documents)}")

    except Exception as e:
        print("ERROR DURING INGESTION OF COARSE CHUNKS:", str(e))    

if(__name__ == "__main__"):
    ingest_coarse_chunks()