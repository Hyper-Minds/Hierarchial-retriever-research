import os
from langchain_qdrant import QdrantVectorStore
import pandas as pd
from uuid6 import uuid7
import json
from vectorstore.set_up_collections import get_qdrant_client
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from ingestion import metadata

load_dotenv()

SUMMARY_ID_TO_CNR_CSV = os.getenv("SUMMARY_ID_TO_CNR_CSV", "C:/Tejeswar/AI Research Engine/Final_Project/data_processed/summary_id_to_cnr_mapping.csv")
COARSE_CHUNKS_QDRANT_COLLECTION = os.getenv("COARSE_CHUNKS_QDRANT_COLLECTION", "qdrant_coarse_chunks_coll_muruga_v4")
COARSE_CHUNKS_2025_FOLDER = os.getenv("COARSE_CHUNKS_2025_FOLDER", "C:/Tejeswar/AI Research Engine/Final_Project/data_processed/coarse_chunks/2025/")
NEW_DATA_FOLDER = os.getenv("NEW_DATA_FOLDER", "C:/Tejeswar/AI Research Engine/Final_Project/data_processed/")
LARGE_EMBEDDING_MODEL_NAME = os.getenv("LARGE_EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")

mapping_df = pd.read_csv(os.path.join(NEW_DATA_FOLDER, SUMMARY_ID_TO_CNR_CSV))

def ingest_coarse_chunks():
    try:
        qdrant_client = get_qdrant_client()

        embedding_model = HuggingFaceEmbeddings(model_name=LARGE_EMBEDDING_MODEL_NAME)

        vector_store = QdrantVectorStore(
                client = qdrant_client,
                collection_name = COARSE_CHUNKS_QDRANT_COLLECTION,
                embedding = embedding_model
            )

        for idx, row in mapping_df.iterrows():
            # process each document 
            summary_id = getattr(row, "summary_id")
            cnr_num = getattr(row, "cnr")

            summary_folder_path = os.path.join(COARSE_CHUNKS_2025_FOLDER, summary_id)
            chunk_files = [f for f in os.listdir(summary_folder_path) if f.endswith(".json")]

            # Stores chunks for a single summary document
            list_of_chunk_documents = []

            for chunk_file in chunk_files:
                # process each chunk file
                with open(os.path.join(summary_folder_path, chunk_file), "r", encoding="utf-8") as f:
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

                    ## If you want metadata, add the metadata field below
                    # This gets metadata dictionary from ingestion/metadata.py
                    metadata_payload = metadata.get_metadata_from_cnr(cnr_num)

                    metadata_payload["doc_id"] = doc_id
                    metadata_payload["chunk_id"] = chunk_id
                    
                    chunk_doc_obj = Document(
                        page_content=chunk_text,
                        metadata = metadata_payload
                    )

                    list_of_chunk_documents.append(chunk_doc_obj)

            vector_store.add_documents(list_of_chunk_documents)
            print("[INGESTION - COARSE CHUNKS]:", idx+1, "/", len(mapping_df), " SUMMARY ID:", summary_id, " TOTAL CHUNKS:", len(list_of_chunk_documents))

    except Exception as e:
        print("ERROR DURING INGESTION OF COARSE CHUNKS:", str(e))    

if(__name__ == "__main__"):
    ingest_coarse_chunks()