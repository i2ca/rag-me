import os
import glob
import pandas as pd
import argparse
from tqdm import tqdm
import traceback
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from gen_model import generate_text, init_model

evaluator_model_id = "meta-llama/Llama-3.1-70B-Instruct"

def _prompt_da_cot_choose(question, system_answer, reasoning, max_score=5):
    prompt = (
        "Given a question, a candidate answer and a professor's evaluation, tell me "
        f"only the final score from 1 to {max_score}.\n\n" 
        "Question:\n"
        f"{question}\n\n"
        "Candidate Answer:\n"
        f"{system_answer}\n\n"
        f"Now based on the following reasoning and evaluation, answer only with the final score from 1 to {max_score}.\n"
        "Reasoning:\n"
        f"{reasoning}\n\n"
        "Answer only with the value of the score and nothing more. Don't try to explain your answer or add any other text."
    )
    return prompt

def process_file(file_path, evaluator_pipeline):
    print(f"Processing file: {file_path}")
    df = pd.read_csv(file_path)
    # Ensure the columns exist (create if necessary)
    if 'da_cot_10_reason' not in df.columns:
        raise Exception(f"Column 'da_cot_10_reason' not found in file {file_path}")
    if 'da_cot_10_score' not in df.columns:
        df['da_cot_10_score'] = None

    for index, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(file_path)):
        # Only process rows where da_cot_10_reason exists and is non-empty
        reasoning = str(row.get('da_cot_10_reason', '')).strip()
        if reasoning:
            question = row.get('question', '')
            system_answer = row.get('system_answer', '')
            prompt = _prompt_da_cot_choose(question, system_answer, reasoning, max_score=10)
            gen_option = generate_text(evaluator_pipeline, prompt, max_new_tokens=5)
            try:
                # Attempt to parse the first one or two characters as integer score.
                try:
                    score = int(gen_option.strip().upper().replace("'", '').replace('"', '').replace("-", '')[:2])
                except Exception:
                    score = int(gen_option.strip().upper().replace("'", '').replace('"', '').replace("-", '')[0])
            except Exception as e:
                print(f"Error parsing score for row {index}: {e}")
                score = -1
            df.at[index, 'da_cot_10_score'] = score
            correct_option = 0
            if score > 6:
                correct_option = 1
            df.at[index, 'da_cot_10'] = correct_option
    # Save the updated DataFrame to the same file
    df.to_csv(file_path, index=False)
    print(f"Updated file saved: {file_path}")

def main(folder, hf_auth):
    # Recursively find all CSV files named "outputs_rag_df_golden_questions.csv" in the given folder
    pattern = os.path.join(folder, '**', '**validation.csv')
    files = glob.glob(pattern, recursive=True)
    if not files:
        print("No files found with the specified name.")
        return

    # Initialize the evaluator model
    try:
        evaluator_pipeline, _, _ = init_model(evaluator_model_id, hf_auth, verbose=True)
    except Exception as e:
        print("Error initializing evaluator model:")
        print(traceback.format_exc())
        return

    # Process each file
    for file_path in files:
        try:
            process_file(file_path, evaluator_pipeline)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files to update da_cot_10_score based on da_cot_10_reason")
    parser.add_argument("--folder", type=str, required=True, help="Folder to search for CSV files")
    parser.add_argument("--hf_auth", type=str, required=True, help="Hugging Face authentication token")
    args = parser.parse_args()
    main(args.folder, args.hf_auth)
