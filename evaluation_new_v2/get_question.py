import os 
from dotenv import load_dotenv

load_dotenv() 

EVAL_DATASET = os.getenv("EVAL_DATASET")
PRIOR_RELEVANCE_JUDGEMENTS = os.getenv("PRIOR_RELEVANCE_JUDGEMENTS")
RELEVENT_JUDGEMENTS_JSON = os.getenv("RELEVENT_JUDGEMENTS_JSON")
EVAL_V2_CASE_DOCS_DIRECTORY = os.getenv("EVAL_V2_CASE_DOCS_DIRECTORY")


def get_question(case_id : str):
    case_id_file = case_id + ".txt"
    