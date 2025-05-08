import gc
import pandas as pd
import sys
import os
import traceback

import torch
from src.ragas_metrics import calculate_ragas_metrics
import src.perplexity as perplexity
import src.classical_metrics as classical_metrics
import src.multiple_choice_eval as multiple_choice_eval
from src.lcad_rag import LcadRag
from dotenv import load_dotenv

load_dotenv()

results_dir =  os.getenv('RESULTS_DIR', "results/default")
embedding_model_name = os.getenv("BERT_SCORE_EMBEDDING_MODEL", "thenlper/gte-large")
model_id = os.getenv("GEN_MODEL", 'mistralai/Mistral-7B-Instruct-v0.2')

#Run evaluation for every file in results_dir
for filename in os.listdir(results_dir):
    if filename.endswith(".csv") and filename != "results_metrics.csv" and not filename.startswith("."):
        try:
            # Load the dataset with questions and answers
            csv_path = os.path.join(results_dir, filename)
            print("\n\nEvaluating ", csv_path)
            try:
                gc.collect()
                torch.cuda.empty_cache()
                multiple_choice_eval.calculate_direct_assessment(csv_path, hf_auth=os.getenv('HUGGINGFACE_AUTH_TOKEN'))
                pass
            except Exception:
                traceback.print_exc()
                print("Could not calculate multiple choice accuracy!")
        except Exception as e:
            traceback.print_exc()
            print("Error: ", str(e))
            print("Error while evaluating: ", csv_path)

#Print results
results_metrics = pd.DataFrame()
list_files = os.listdir(results_dir)
list_files.sort()
for filename in list_files:
    if filename.endswith(".csv") and filename != "results_metrics.csv":
        csv_path = os.path.join(results_dir, filename)
        results = pd.read_csv(csv_path)
        da_cot_10 = results['da_cot_10'].mean() if 'da_cot_10' in results.columns else float('nan')

        metrics_dict = {
                'filename': [csv_path], 
                'da_cot_10': [da_cot_10]
                }
        if results_metrics.empty:
            results_metrics = pd.DataFrame(metrics_dict)
        else:
            results_metrics = pd.concat([results_metrics, 
                                        pd.DataFrame(metrics_dict)], ignore_index=True)
        
print(results_metrics.to_string())
results_metrics.to_csv(results_dir+'/results_metrics.csv', index=False)