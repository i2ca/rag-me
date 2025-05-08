import pandas as pd
import numpy as np
import json
import random

# Read csv
df_questions = pd.read_csv("../data/cvpr-papers-500/df_golden_questions_test.csv")

# Mapping DataFrame to desired JSON format
output = []
for index, row in df_questions.iterrows():
    context = row['text']
    question = row['question']
    correct_answer = row['correct_answer']
    wrong_answers = [row['wrong_answer1'], row['wrong_answer2'], row['wrong_answer3']]
    
    options = wrong_answers + [correct_answer]
    random.shuffle(options)
    
    correct_answer_index = options.index(correct_answer)
    
    answer_key = chr(97 + correct_answer_index)  # Converts 0 to 'a', 1 to 'b', etc.
    
    entry = {
        "context": context,
        "question": {
            "statement": question,
            "options": options,
            "answer": answer_key
        }
    }
    output.append(entry)

# Convert to JSON and save to a file
with open('golden_questions.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print("JSON file has been created.")