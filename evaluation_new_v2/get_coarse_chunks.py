
from dotenv import load_dotenv
import os 
import json
from dotenv import load_dotenv
from uuid6 import uuid7
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re 
import shutil

load_dotenv()

# NEW_DATA_2025_TEXT_FOLDER = os.getenv("NEW_DATA_2025_TEXT_FOLDER")
# COARSE_CHUNKS_2025_FOLDER = os.getenv("COARSE_CHUNKS_2025_FOLDER")
# NEW_DATA_FOLDER = os.getenv("NEW_DATA_FOLDER")
# SUMMARY_ID_TO_CNR_CSV = os.getenv("SUMMARY_ID_TO_CNR_CSV")

EVAL_V2_CASE_DOCS_DIRECTORY = os.getenv("EVAL_V2_CASE_DOCS_DIRECTORY")
EVAL_V2_COARSE_CHUNKS_DIRECTORY = os.getenv("EVAL_V2_COARSE_CHUNKS_DIRECTORY")
EVAL_DATASET = os.getenv("EVAL_DATASET")
EVAL_V2_RELEVENT_JUDGEMENTS_JSON = os.getenv("RELEVENT_JUDGEMENTS_JSON_NAME")
EVAL_DIR_V2 = os.getenv("EVAL_DIR_V2")
EVAL_V2_RELEVENT_JUDGEMENT_LIST_JSON = os.getenv("EVAL_V2_RELEVENT_JUDGEMENT_LIST_JSON")

SECTION_PATTERNS = {
    "case_details": r"(case\s+details)",
    "issues": r"(issues?\s+(for\s+consideration|involved))",
    "headnotes": r"(head\s*notes?)",
    "citations": r"(citations?|case law cited)",
    "acts" : r"(list of acts?)",
    "keywords": r"(list of keywords?|keywords)",
    "arising" : r"((case\s+arising\s+out\s+of)|(case\s+arising\s+from))",
    "appearances": r"(appearances\s+for\s+parties)",
    "judgment": r"(order of the supreme court|judgement)"
}

SECTION_ORDER = ["case_details", "issues", "headnotes", "citations", "acts", "keywords", "other_details", "judgment"]

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"Digital Supreme Court Reports", "", text)
    return text.strip()

def split_into_sections(text: str):
    text = normalize_text(text)

    matches = []
    for section, pattern in SECTION_PATTERNS.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append({
                "section": section,
                "start": match.start()
            })

    if not matches:
        return [{"section": "unknown", "text": text}]

    matches = sorted(matches, key=lambda x: x["start"])

    sections = []
    for i, current in enumerate(matches):
        start = current["start"]
        end = matches[i + 1]["start"] if i + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()
        sections.append({
            "section": current["section"],
            "text": section_text
        })

    return sections


def split_section_into_chunks(section_text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(section_text)


def hybrid_chunk_document(doc_id, text, output_dir):
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

def store_coarse_chunks():

    if(os.path.exists(EVAL_V2_COARSE_CHUNKS_DIRECTORY) == False):
        os.makedirs(EVAL_V2_COARSE_CHUNKS_DIRECTORY)

    # remove the contents
    shutil.rmtree(EVAL_V2_COARSE_CHUNKS_DIRECTORY)

    with open(os.path.join(EVAL_DIR_V2, EVAL_V2_RELEVENT_JUDGEMENTS_JSON), "r", encoding="utf-8") as f:
        relevent_judgements = json.load(f)

    chunked_document_id = set()
    total_docs = 0
    total_chunks = 0
    
    for question_id in relevent_judgements:
        relevent_case_id_to_q_list = relevent_judgements[question_id]

        for relevent_case_id in relevent_case_id_to_q_list:

            # check if the document os already chunked
            if relevent_case_id in chunked_document_id:
                continue

            chunked_document_id.add(relevent_case_id)
            total_docs += 1


            relevent_case_file = relevent_case_id + ".txt"
            text_file_path = os.path.join(EVAL_V2_CASE_DOCS_DIRECTORY, relevent_case_file)
            
            # read the text file 
            with open(text_file_path, 'r', encoding="utf-8") as f:
                document_text = f.read()  

            CHUNK_OUTPUT_DIR = os.path.join(EVAL_V2_COARSE_CHUNKS_DIRECTORY, relevent_case_id)
            # print(CHUNK_OUTPUT_DIR)

            chunk_id_lists = hybrid_chunk_document(relevent_case_id, document_text, CHUNK_OUTPUT_DIR)
            total_chunks += len(chunk_id_lists)

            print(f"CHUNKING DONE FOR DOCUMENT: {relevent_case_id} , Number of chunks: {len(chunk_id_lists)}")
 
    relevent_judgement_id_list = list(chunked_document_id)
    # store the doc id in JSON for easy reading 
 
    with open(os.path.join(EVAL_DIR_V2, EVAL_V2_RELEVENT_JUDGEMENT_LIST_JSON), 'w', encoding="utf-8") as file:
        json.dump({"relevent_docs_id" : relevent_judgement_id_list}, file)

    print(f"Total Documents: {total_docs} , Total Chunks: {total_chunks}")

if(__name__ == "__main__"):
    store_coarse_chunks()