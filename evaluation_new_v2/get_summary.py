import os 
import pandas as pd 
from dotenv import load_dotenv
from llm import llama, prompt
from uuid6 import uuid7
import sys 


load_dotenv() 
print(1) 
EVAL_V2_COARSE_CHUNK_COLLECTION_NAME = os.getenv("EVAL_V2_COARSE_CHUNK_COLLECTION_NAME")
EVAL_V2_SUMMARY_COLLECTION_NAME = os.getenv("EVAL_V2_SUMMARY_COLLECTION_NAME")

EVAL_DIR_V2 = os.getenv("EVAL_DIR_V2")
EVAL_V2_COARSE_CHUNKS_DIRECTORY= os.getenv("EVAL_V2_COARSE_CHUNKS_DIRECTORY")

EVAL_V2_RELEVENT_JUDGEMENT_LIST_JSON = os.getenv("EVAL_V2_RELEVENT_JUDGEMENT_LIST_JSON")
LARGE_EMBEDDING_MODEL_NAME = os.getenv("LARGE_EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")

if(not (os.path.exists(SUMMARY_OUTPUT_2025_FOLDER))):
    os.makedirs(SUMMARY_OUTPUT_2025_FOLDER)

print(metadata_df.head())

summary_id_to_cnr = {}

def save_mappings(summary_id_to_cnr):
    summary_mapping_df = pd.DataFrame(list(summary_id_to_cnr.items()), columns=["summary_id", "cnr"])
    summary_mapping_df.to_csv(os.path.join(NEW_DATA_FOLDER, "summary_id_to_cnr_mapping.csv"), index=False, quoting=1)

case_count = 0
for row in metadata_df.itertuples():

    try:
        case_id = getattr(row, "cnr")
        text_file_path = os.path.join(NEW_DATA_2025_TEXT_FOLDER, f"{case_id}.txt")
        
        with open(text_file_path, "r", encoding="utf-8") as text_file:
            document_text = text_file.read()
        
        metadata = {
            "title": getattr(row, "title"),
            "court": getattr(row, "court"),
            "respondent": getattr(row, "respondent"),
            "petitioner": getattr(row, "petitioner"),
            "judge": getattr(row, "judge"),
            "author_judge": getattr(row, "author_judge"),
            "cnr": getattr(row, "cnr"),
            "citation": getattr(row, "citation"),
            "disposal_nature": getattr(row, "disposal_nature"),
            "decision_date": getattr(row, "decision_date"),
            "year": getattr(row, "year")
        }
        summary_prompt = prompt.get_summary_prompt(metadata, document_text) 
        summary = llama.get_answer(summary_prompt)

        summary_id = str(uuid7())
        summary_id_to_cnr[summary_id]  = case_id

        summary_file_path =  os.path.join(SUMMARY_OUTPUT_2025_FOLDER, f"{summary_id}.txt")
        with open(summary_file_path, "w", encoding="utf-8") as summary_file:
            summary_file.write(summary)

        case_count += 1
        print(f"Processed Summary for Case ID: {case_id} count : {case_count} / 50" )

    except EOFError:
        # store the dict summary_id_to_cnr as a csv file
        save_mappings(summary_id_to_cnr)
        print(f"EOFError encountered for Case ID: {case_id}. Skipping this case.")
        sys.exit(0)    
    
    except KeyboardInterrupt:
        # store the dict summary_id_to_cnr as a csv file
        save_mappings(summary_id_to_cnr)
        print(f"Process interrupted by user at Case ID: {case_id}. Saving progress and exiting.")
        sys.exit(0)    

save_mappings(summary_id_to_cnr)
print("All summaries processed and mappings saved.")    