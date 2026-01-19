
from dotenv import load_dotenv
import os 
import pandas as pd
import json
from dotenv import load_dotenv
import pandas as pd 
from uuid6 import uuid7
from langchain_text_splitters import RecursiveCharacterTextSplitter
from extraction import store_coarse_chunks
from extraction.sections import split_into_sections

load_dotenv()

NEW_DATA_2025_TEXT_FOLDER = os.getenv("NEW_DATA_2025_TEXT_FOLDER")
NEW_DATA_FOLDER = os.getenv("NEW_DATA_FOLDER")
SUMMARY_ID_TO_CNR_CSV = os.getenv("SUMMARY_ID_TO_CNR_CSV")
NORMAL_CHUNKS_2025_FOLDER = os.getenv("NORMAL_CHUNKS_2025_FOLDER")

def split_section_into_chunks(section_text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(section_text)


def get_normal_chunks(doc_id, text, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    sections = split_into_sections(text)
    all_chunks = []

    for sec in sections:
        section_name = sec["section"]
        section_text = sec["text"]

        chunks = split_section_into_chunks(section_text)

        for chunk_text in chunks:
            chunk_id = str(uuid7())

            chunk_obj = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "section": section_name,
                "text": chunk_text
            }    

            with open(os.path.join(output_dir, f"{chunk_id}.json"), "w", encoding="utf-8") as f:
                json.dump(chunk_obj, f, ensure_ascii=False, indent=2)

            all_chunks.append(chunk_id)

    return all_chunks

def store_normal_chunks():

    if(os.path.exists(NORMAL_CHUNKS_2025_FOLDER) == False):
        os.makedirs(NORMAL_CHUNKS_2025_FOLDER)

    df = pd.read_csv(os.path.join(NEW_DATA_FOLDER, SUMMARY_ID_TO_CNR_CSV))
    
    documents_count = len(df)

    for idx, row in df.iterrows():
        document_id = getattr(row, "summary_id")
        cnr_num = getattr(row, "cnr")

        text_file_path = os.path.join(NEW_DATA_2025_TEXT_FOLDER, f"{cnr_num}.txt")

        total_chunks = []
    
        with open(text_file_path, "r", encoding="utf-8") as text_file:
            document_text = text_file.read()  

        CHUNK_OUTPUT_DIR = os.path.join(NORMAL_CHUNKS_2025_FOLDER, document_id)

        chunk_lists = get_normal_chunks(document_id, document_text, CHUNK_OUTPUT_DIR)
        total_chunks.extend(chunk_lists)

        print("CHUNKING DONE FOR DOCUMENT:", idx+1, "/", documents_count, " CNR:", cnr_num, " DOC ID:", document_id)

    return total_chunks


if(__name__ == "__main__"):
    store_normal_chunks()