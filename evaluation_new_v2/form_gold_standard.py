import os 
import pandas as pd 
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv() 

EVAL_DATASET_FIRE = os.getenv("EVAL_DATASET", "../FIRE 2029 eval dataset")
PRIOR_RELEVANCE_JUDGEMENTS = os.getenv("PRIOR_RELEVANCE_JUDGEMENTS", "../FIRE 2029 eval dataset/relevance_judgments_priorcases.txt")

ground_truth_judgements = defaultdict(list)

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
                print(judgement_id)

            


if(__name__ == "__main__"):
    get_ground_truth()