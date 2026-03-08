import os 
import pandas as pd 
from dotenv import load_dotenv
from collections import defaultdict
import json 

load_dotenv() 

EVAL_DATASET = os.getenv("EVAL_DATASET")
PRIOR_RELEVANCE_JUDGEMENTS = os.getenv("PRIOR_RELEVANCE_JUDGEMENTS")
RELEVENT_JUDGEMENTS_JSON = os.getenv("RELEVENT_JUDGEMENTS_JSON_NAME")
EVAL_DIR_V2 = os.getenv("EVAL_DIR_V2")

ground_truth_judgements = defaultdict(set)
total_relevant_docs = 0

def get_ground_truth():
    with open(PRIOR_RELEVANCE_JUDGEMENTS, encoding="utf-8" ) as file:
        for row in file:
            # print(row)
            row_list = row.split()
    
            if(len(row_list) < 4):
                continue

            relevancy = row_list[3]
            judgement_id = row_list[2]
            question_id = row_list[0]

            if(relevancy == "1"):
                ground_truth_judgements[question_id].add(judgement_id) 

        # convert the set to list 
        for key in ground_truth_judgements:
            ground_truth_judgements[key] = list(ground_truth_judgements[key])

        with open(os.path.join(EVAL_DIR_V2, RELEVENT_JUDGEMENTS_JSON), "w", encoding="utf-8") as f:
            json.dump(ground_truth_judgements, f, ensure_ascii=False, indent=2)




if(__name__ == "__main__"):
    get_ground_truth()