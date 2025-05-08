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

# rag = LcadRag(
#     model_id=model_id, 
#     encoder_model_id=embedding_model_name,
#     data_dir=os.getenv("DATA_DIR", "data/default"),
# )

#Run evaluation for every file in results_dir
for filename in os.listdir(results_dir):
    if filename.endswith(".csv") and filename != "results_metrics.csv" and not filename.startswith("."):
        try:
            # Load the dataset with questions and answers
            csv_path = os.path.join(results_dir, filename)
            print("\n\nEvaluating ", csv_path)
            # classical_metrics.calculate_bert_score(rag.sentence_retriever.model, csv_path)
            # classical_metrics.calculate_rouge_score(csv_path)
            try:
                # perplexity.calculate_perplexity(rag, csv_path)
                # perplexity.calculate_perplexity_accuracy(rag, csv_path)
                # perplexity.calculate_perplexity_accuracy_options(rag, csv_path)
                pass
            except Exception:
                print("Could not calculate perplexity! Ensure you are using a local model.")
            try:
                # multiple_choice_eval.calculate_multiple_choice_accuracy(rag, csv_path)
                # del rag
                gc.collect()
                torch.cuda.empty_cache()
                # multiple_choice_eval.calculate_mca_v2(csv_path, hf_auth=os.getenv('HUGGINGFACE_AUTH_TOKEN'))
                # multiple_choice_eval.calculate_mca_v2_chain_of_thought(csv_path, hf_auth=os.getenv('HUGGINGFACE_AUTH_TOKEN'))
                multiple_choice_eval.calculate_direct_assessment(csv_path, hf_auth=os.getenv('HUGGINGFACE_AUTH_TOKEN'))
                # multiple_choice_eval.calculate_direct_assessment_examples(csv_path, hf_auth=os.getenv('HUGGINGFACE_AUTH_TOKEN'))
                # multiple_choice_eval.calculate_direct_assessment_steps(csv_path, hf_auth=os.getenv('HUGGINGFACE_AUTH_TOKEN'))
                # multiple_choice_eval.calculate_direct_assessment_ragas(csv_path, hf_auth=os.getenv('HUGGINGFACE_AUTH_TOKEN'))
                # multiple_choice_eval.calculate_mca_with_direct_assessment(csv_path, hf_auth=os.getenv('HUGGINGFACE_AUTH_TOKEN'))
                # rag = LcadRag(model_id=model_id, encoder_model_id=embedding_model_name, data_dir=os.getenv("DATA_DIR", "data/default"))
                pass
            except Exception:
                traceback.print_exc()
                print("Could not calculate multiple choice accuracy!")
            try:
                # Calculate Ragas metrics - This metric need an OpenAI API key
                # calculate_ragas_metrics(rag.model_pipeline, csv_path)
                pass
            except Exception:
                traceback.print_exc()
                print("Could not calculate ragas!")
            # ragas.calculate_faithfulness(rag.model_pipeline, csv_path, calculate_context=("rag_True" in filename))
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
        bert_score = results['bert_score'].mean() if 'bert_score' in results.columns else float('nan')
        rouge_score = results['rouge_score'].mean() if 'rouge_score' in results.columns else float('nan')
        ppl = results['perplexity'].mean() if 'perplexity' in results.columns else float('nan')
        ppl_accuracy = results['perplexity_accuracy'].mean() if 'perplexity_accuracy' in results.columns else float('nan')
        ppl_accuracy_options = results['perplexity_accuracy_options'].mean() if 'perplexity_accuracy_options' in results.columns else float('nan')
        answer_relevance = results['answer_relevance_score'].mean() \
                            if 'answer_relevance_score' in results.columns else float('nan')
        faithfulness_text = results['faithfulness_text_eval_score'].mean() \
                            if 'faithfulness_text_eval_score' in results.columns else float('nan')
        faithfulness_context = results['faithfulness_context_eval_score'].mean() \
                            if 'faithfulness_context_eval_score' in results.columns else float('nan')
        multiple_choice_accuracy = results['multiple_choice_accuracy'].mean() \
                            if 'multiple_choice_accuracy' in results.columns else float('nan')
        mca_v2 = results['mca_v2'].mean() if 'mca_v2' in results.columns else float('nan')
        mca_v2_no_context = results['mca_v2_no_context'].mean() if 'mca_v2_no_context' in results.columns else float('nan')
        mca_v2_cot = results['mca_v2_cot'].mean() if 'mca_v2_cot' in results.columns else float('nan')
        mca_v2_cot_no_context = results['mca_v2_cot_no_context'].mean() if 'mca_v2_cot_no_context' in results.columns else float('nan')
        da_cot_10 = results['da_cot_10'].mean() if 'da_cot_10' in results.columns else float('nan')

        metrics_dict = {
                'filename': [csv_path], 
                'bert_score': [bert_score], 
                'rouge_score': [rouge_score], 
                #'faithfulness_context': [faithfulness_context],
                #'faithfulness_text': [faithfulness_text], 
                # 'perplexity': [ppl], 
                'perplexity_accuracy': [ppl_accuracy],
                # 'perplexity_accuracy_options': [ppl_accuracy_options],
                'multiple_choice_accuracy': [multiple_choice_accuracy],
                # 'mca_v2': [mca_v2],
                # 'mca_v2_no_context': [mca_v2_no_context],
                # 'mca_v2_cot': [mca_v2_cot],
                'mca_v2_cot_no_context': [mca_v2_cot_no_context],
                'da_cot_10': [da_cot_10]
                }
        if results_metrics.empty:
            results_metrics = pd.DataFrame(metrics_dict)
        else:
            results_metrics = pd.concat([results_metrics, 
                                        pd.DataFrame(metrics_dict)], ignore_index=True)
        
print(results_metrics.to_string())
results_metrics.to_csv(results_dir+'/results_metrics.csv', index=False)