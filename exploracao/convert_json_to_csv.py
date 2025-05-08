import pandas as pd
import numpy as np
import json

# Read JSON file
with open('golden_questions_valid.json', 'r') as f:
    data = json.load(f)
    df_questions = pd.DataFrame(columns=['question', 'correct_answer', 'wrong_answer1', 'wrong_answer2', 'wrong_answer3', 'text'])
    for entry in data:
        context = entry['context']
        question = entry['question']['statement']
        answer_key = entry['question']['answer']
        options = entry['question']['options']

        char_to_int = {"a": 0, "b": 1, "c": 2, "d": 3}

        correct_answer = options[char_to_int[answer_key]]
        wrong_answers = options[:char_to_int[answer_key]] + options[char_to_int[answer_key] + 1:]

        # Create DataFrame
        df = pd.DataFrame({
            'question': [str(question)],
            'correct_answer': [str(correct_answer)],
            'wrong_answer1': [str(wrong_answers[0])],
            'wrong_answer2': [str(wrong_answers[1])],
            'wrong_answer3': [str(wrong_answers[2])],
            'text': [str(context)],
        })

        df_questions = pd.concat([df_questions, df], ignore_index=True)


    # Save DataFrame to CSV
    df_questions.to_csv(f'../data/cvpr-papers-500/df_golden_questions_test_valid.csv', index=False)
        
