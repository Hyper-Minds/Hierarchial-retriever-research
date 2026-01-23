import os
from langchain_qdrant import QdrantVectorStore
import pandas as pd
from vectorstore.set_up_collections import get_qdrant_client
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from ingestion import metadata

load_dotenv()

SUMMARY_ID_TO_CNR_CSV = os.getenv("SUMMARY_ID_TO_CNR_CSV", "C:/Tejeswar/AI Research Engine/Final_Project/data_processed/summary_id_to_cnr_mapping.csv")
SUMMARY_OUTPUT_2025_FOLDER = os.getenv("SUMMARY_OUTPUT_2025_FOLDER", "C:/Tejeswar/AI Research Engine/Final_Project/data_processed/summaries/2025/")
NEW_DATA_FOLDER = os.getenv("NEW_DATA_FOLDER", "C:/Tejeswar/AI Research Engine/Final_Project/data_processed/")

SUMMARY_QDRANT_COLLECTION = os.getenv("SUMMARY_QDRANT_COLLECTION", "qdrant_summary_coll_muruga_v4")

LARGE_EMBEDDING_MODEL_NAME = os.getenv("LARGE_EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")

mapping_df = pd.read_csv(os.path.join(NEW_DATA_FOLDER, SUMMARY_ID_TO_CNR_CSV))

def ingest_summaries():
    qdrant_client = get_qdrant_client()

    embedding_model = HuggingFaceEmbeddings(model_name=LARGE_EMBEDDING_MODEL_NAME)

    vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=SUMMARY_QDRANT_COLLECTION,
            embedding=embedding_model  
        )

    for idx, row in mapping_df.iterrows():
        summary_id = getattr(row, "summary_id")
        cnr_num = getattr(row, "cnr")

        summary_file_path = os.path.join(SUMMARY_OUTPUT_2025_FOLDER, f"{summary_id}.txt")
        
        with open(summary_file_path, "r", encoding="utf-8") as summary_file:
            summary_text = summary_file.read()

        # If you want metadata, add the metadata field below
        # This gets metadata dictionary from ingestion/metadata.py
        metadata_payload = metadata.get_metadata_from_cnr(cnr_num)

        # Set the payload fields in the Document metadata
        metadata_payload["summary_id"] = summary_id
        metadata_payload["cnr"] = cnr_num

        document_summary_obj = Document(
            page_content = summary_text,
            metadata = metadata_payload
        )

        vector_store.add_documents([document_summary_obj])
        print("[INGESTION - SUMMARY DOCS]:", idx+1, "/", len(mapping_df), " CNR:", cnr_num, " SUMMARY ID:", summary_id)

if(__name__ == "__main__"):
    ingest_summaries()