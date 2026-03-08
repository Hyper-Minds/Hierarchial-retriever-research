import os 
import pandas as pd

NEW_DATA_2025_METADATA_FOLDER = "C:/Tejeswar/AI Research Engine/Final_Project/data_processed/metadata/2025"
NEW_DATA_2025_METADATA_CSV_NAME = "metadata_2025.csv"

NEW_DATA_FOLDER = "C:/Tejeswar/AI Research Engine/Final_Project/data_processed/"
SUMMARY_ID_TO_CNR_CSV = "summary_id_to_cnr_mapping.csv"

metadata_fields_needed = ["court", "year", "case_id", "petitioner", "respondent", "judge", "case_num", "cnr" , "path", "disposal_nature"]

def get_metadata_fields_needed():
    return metadata_fields_needed

def get_metadata_from_cnr(cnr_number):
    mapping_df = pd.read_csv(os.path.join(NEW_DATA_2025_METADATA_FOLDER, NEW_DATA_2025_METADATA_CSV_NAME))

    # Change the NaN values in the dataframe to " "
    mapping_df = mapping_df.fillna(" ")

    metadata_row_dict = mapping_df[mapping_df["cnr"] == cnr_number].to_dict(orient="records")

    payload = {}

    if(len(metadata_row_dict) == 0):
        print("CNR NUMBER NOT FOUND IN METADATA:", cnr_number)
        metadata_row_dict = {}
    else:
        # Get the first matching row
        metadata_row_dict = metadata_row_dict[0]

    for key in metadata_fields_needed:
        payload[key] = metadata_row_dict.get(key, None)

    payload["year"] = str(payload["year"])  # Ensure year is string

    # print(payload)
    return payload 

def get_metadata_from_summary_id(summary_id):
    mapping_df = pd.read_csv(os.path.join(NEW_DATA_FOLDER, SUMMARY_ID_TO_CNR_CSV))

    # Change the NaN values in the dataframe to " "
    mapping_df = mapping_df.fillna(" ")

    cnr_number = mapping_df[mapping_df["summary_id"] == summary_id]["cnr"].item()

    return get_metadata_from_cnr(cnr_number)

if(__name__ == "__main__"):
    cnr_num = "ESCR010000022025"
    get_metadata_from_cnr(cnr_num)
        