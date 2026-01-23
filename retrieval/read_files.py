import os 
from dotenv import load_dotenv
import json

load_dotenv()

NEW_DATA_FOLDER = os.getenv("NEW_DATA_FOLDER", "C:/Tejeswar/AI Research Engine/Final_Project/data_processed/")
SUMMARY_OUTPUT_2025_FOLDER = os.getenv("SUMMARY_OUTPUT_2025_FOLDER", "C:/Tejeswar/AI Research Engine/Final_Project/data_processed/summaries/2025/")
COARSE_CHUNKS_2025_FOLDER = os.getenv("COARSE_CHUNKS_2025_FOLDER", "C:/Tejeswar/AI Research Engine/Final_Project/data_processed/coarse_chunks/2025/")

def read_summary_file(summary_id):
    summary_text = ""
    summary_file_path = os.path.join(SUMMARY_OUTPUT_2025_FOLDER, f"{summary_id}.txt")

    if(os.path.exists(summary_file_path)):
        with open(summary_file_path, "r", encoding="utf-8") as summary_file:
            summary_text = summary_file.read()

    return summary_text

def read_coarse_chunk_file(summary_id, chunk_id):
    chunk_text = ""
    chunk_file_path = os.path.join(COARSE_CHUNKS_2025_FOLDER, summary_id, f"{chunk_id}.json")

    if(os.path.exists(chunk_file_path)):
        with open(chunk_file_path, "r", encoding="utf-8") as chunk_file:
            chunk_text = json.load(chunk_file)

    return chunk_text  

if(__name__ == "__main__"):
    # Test reading a summary file
    test_summary_id = "019ba0cf-9e46-7622-8bb5-241e4f244cc9"
    chunk_id = "019be27a-816a-7104-b149-e532761d5b90"
    chunk_content = read_coarse_chunk_file(test_summary_id, chunk_id)
    print("Chunk Content:", chunk_content) 