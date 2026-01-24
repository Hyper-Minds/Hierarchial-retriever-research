import os 
import pandas as pd 
from dotenv import load_dotenv
from llm import llama, prompt
from uuid6 import uuid7
import sys 

load_dotenv() 
print(1) 

SUMMARY_OUTPUT_2025_FOLDER = os.getenv("SUMMARY_OUTPUT_2025_FOLDER")

NEW_DATA_2025_TEXT_FOLDER = os.getenv("NEW_DATA_2025_TEXT_FOLDER")
NEW_DATA_2025_METADATA_FOLDER = os.getenv("NEW_DATA_2025_METADATA_FOLDER")
NEW_DATA_2025_METADATA_CSV_NAME = os.getenv("NEW_DATA_2025_METADATA_CSV_NAME")
NEW_DATA_FOLDER = os.getenv("NEW_DATA_FOLDER")

if(not (os.path.exists(SUMMARY_OUTPUT_2025_FOLDER))):
    os.makedirs(SUMMARY_OUTPUT_2025_FOLDER)

metadata_df = pd.read_csv(os.path.join(NEW_DATA_2025_METADATA_FOLDER, NEW_DATA_2025_METADATA_CSV_NAME))
old_cnr_to_doc_id = pd.read_csv("C:/Tejeswar/AI Research Engine/Final_Project/v1_data_processed/summary_id_to_cnr_mapping.csv")

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
        
        summary_id = old_cnr_to_doc_id[old_cnr_to_doc_id["cnr"] == case_id]["summary_id"].values[0]
        print(summary_id)

        old_summary_file_path = os.path.join("C:/Tejeswar/AI Research Engine/Final_Project/v1_data_processed/summaries/2025", f"{summary_id}.txt")

        with open(old_summary_file_path, "r", encoding="utf-8") as text_file:
            summary_text = text_file.read()

        summary_id_to_cnr[summary_id]  = case_id

        # write the summary text to new folder with same name 
        new_summary_file_path = os.path.join(SUMMARY_OUTPUT_2025_FOLDER, f"{summary_id}.txt")
        with open(new_summary_file_path, "w", encoding="utf-8") as summary_file:
            summary_file.write(summary_text)

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