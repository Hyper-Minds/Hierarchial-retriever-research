import os
import pandas as pd 
import numpy as np
from dotenv import load_dotenv 
from extraction.pdf_utils import extractPDF
import re

SCR_HEADER_PATTERN = re.compile(
    r"^\[\d{4}\]\s*\d+\s*S\.C\.R\.?\s*$",
    re.IGNORECASE
)

DSCR_PATTERN = re.compile(
    r"^Digital\s+Supreme\s+Court\s+Reports\s*$",
    re.IGNORECASE
)

def remove_page_headers(text: str) -> str:
    cleaned_lines = []

    for line in text.splitlines():
        stripped = line.strip()

        if SCR_HEADER_PATTERN.match(stripped):
            continue

        if DSCR_PATTERN.match(stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


load_dotenv()

OLD_DATA_2025_PDF = os.getenv("OLD_DATA_2025_PDF")
OLD_DATA_2025_METADATA_FILE = os.getenv("OLD_DATA_2025_METADATA_FILE")

NEW_DATA_2025_TEXT_FOLDER = os.getenv("NEW_DATA_2025_TEXT_FOLDER")
NEW_DATA_2025_METADATA_FOLDER = os.getenv("NEW_DATA_2025_METADATA_FOLDER")
NEW_DATA_2025_METADATA_CSV_NAME = os.getenv("NEW_DATA_2025_METADATA_CSV_NAME")

if(os.path.exists(NEW_DATA_2025_METADATA_FOLDER) == False):
    os.makedirs(NEW_DATA_2025_METADATA_FOLDER)

if(os.path.exists(NEW_DATA_2025_TEXT_FOLDER) == False):
    os.makedirs(NEW_DATA_2025_TEXT_FOLDER)

not_needed_metadata_keys = ["raw_html", "available_languages", "scraped_at"]

def extract_metadata():
    """
    Extract 50 PDFs from the data folder and store it in a separate folder.
    """
    metadata_df = pd.read_parquet(OLD_DATA_2025_METADATA_FILE)
    metadata_df = metadata_df.replace({pd.NA: None, np.nan: None})
    metadata_df = metadata_df.rename(columns={"nc_display" : "neutral_citation"})
    metadata_df["path"] = metadata_df["path"] + "_EN"

    selected_df = metadata_df.sort_values(by = ["decision_date", "cnr"], ascending=True)
    selected_df = selected_df.drop_duplicates(subset=["cnr"])
    selected_df = selected_df.drop(columns=not_needed_metadata_keys)

    print(selected_df.head())

    num_cases_to_process = 50

    processed_cases = 0
    seen_cases = set()

    metadata_columns = list(selected_df.columns)
    quoted_metadata_columns = ['\"' + col + '\"' for col in metadata_columns]
    

    # write the header to csv file

    with open(os.path.join(NEW_DATA_2025_METADATA_FOLDER, NEW_DATA_2025_METADATA_CSV_NAME), "w", encoding="utf-8") as metadata_csv:
        metadata_csv.write(",".join(quoted_metadata_columns) + "\n")
    # selected_df.to_csv(os.path.join(NEW_DATA_2025_METADATA_FOLDER, NEW_DATA_2025_METADATA_CSV_NAME), index=False, quoting=1)

    # iterate each row in dataframe
    for row in selected_df.itertuples():
        if processed_cases >= num_cases_to_process:
            break

        case_id = getattr(row, "cnr")
        if case_id in seen_cases:
            continue

        seen_cases.add(case_id)
        processed_cases += 1

        # write metadata row to csv file
        with open(os.path.join(NEW_DATA_2025_METADATA_FOLDER, NEW_DATA_2025_METADATA_CSV_NAME), "a", encoding="utf-8") as metadata_csv:
            row_values = ['\"' + str(getattr(row, col)) + '\"' if getattr(row, col) is not None else "" for col in metadata_columns]
            metadata_csv.write(",".join(row_values) + "\n")

        pdf_path = getattr(row, "path") + ".pdf"
        full_pdf_path = os.path.join(OLD_DATA_2025_PDF, pdf_path)

        # Extract text from PDF
        pdf_text = extractPDF(full_pdf_path)
        cleaned_text = remove_page_headers(pdf_text)

        if cleaned_text == '':
            continue  

        # Save extracted text to a file
        text_file_name = case_id + ".txt"
        text_file_path = os.path.join(NEW_DATA_2025_TEXT_FOLDER, text_file_name)
        with open(text_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(cleaned_text)
        print(f"Processed case {case_id} : {processed_cases} out of 100.")



if __name__ == "__main__":
    extract_metadata() 