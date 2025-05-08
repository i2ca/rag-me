import gradio as gr
import pandas as pd
from src.lcad_rag import LcadRag
from tqdm import tqdm
import traceback
import os
from dotenv import load_dotenv

load_dotenv()

# FLAGS FOR DETERMINING WHAT THIS SCRIPT WILL DO
start_gradio = False #Only change this
generate_answers = not start_gradio
# Change this to inform where the RAG data is and where the results should be saved inside the dir evaluation/
data_dir = os.getenv('DATA_DIR', 'data/default')
results_dir = os.getenv('RESULTS_DIR', 'results/default')
model_id = os.getenv("GEN_MODEL", 'mistralai/Mistral-7B-Instruct-v0.3')

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "None")
EMBEDDING_MODEL_DIMENSIONS = os.getenv("EMBEDDING_MODEL_DIMENSIONS", 1024)
BERT_SCORE_EMBEDDING_MODEL = os.getenv("BERT_SCORE_EMBEDDING_MODEL", "thenlper/gte-large")
GEN_MODEL = model_id
CHUNK_SIZE = os.getenv("CHUNK_SIZE", 1000)
OVERLAP_CHUNKS = os.getenv("OVERLAP_CHUNKS", 80)

if not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok=True)

if start_gradio:
    model = LcadRag(model_id=GEN_MODEL, rag_on=True, rag_use_text=True, rag_use_questions=False, \
                    rag_use_summary=False, rag_query_expansion=False, data_dir=data_dir, verbose=True)
    gr.ChatInterface(model.get_model_response).launch(server_port=7860, share=True)
else:
    def generate_answers_from_dataset(model: LcadRag, df_questions_answers, output_name="outputs.csv"):
        print('Generating answers:')
        print(output_name)
        
        try:
            df_questions_answers["system_answer"] = ""
            df_questions_answers["rag_context"] = ""
            df_questions_answers["llm_response"] = ""
            df_questions_answers.drop(columns=["llm_response"], inplace=True)
            for idx in tqdm(range(len(df_questions_answers.index))): 
                row = df_questions_answers.iloc[idx]
                question = row['question']
                response, rag_context = model.get_model_response_and_context(question, history=[])

                #LOGS
                #print(f"\n--------\n\nQuestion: {question}\n")
                #print(f"Rag context: {rag_context}")
                #print(f"Response: {response}")

                df_questions_answers.at[idx, 'system_answer'] = response
                df_questions_answers.at[idx, 'rag_context'] = rag_context

                if idx % 10 == 0:
                    csv_path = f'{results_dir}/{output_name}'
                    # print(f"Saving to {csv_path}")
                    df_questions_answers.to_csv(csv_path, index=False)
        except Exception as e:
            print("Error: ", str(e))
            traceback.print_exc()
            print("Error while generating answers: ", filename)
            print("Saving current progress...")
        csv_path = f'{results_dir}/{output_name}'
        print(f"Saving to {csv_path}")
        df_questions_answers.to_csv(csv_path, index=False)

    model = LcadRag(model_id=model_id, rag_on=True, rag_use_text=True, rag_use_questions=False, \
                                rag_use_summary=False, rag_query_expansion=False, data_dir=data_dir)
    if generate_answers:
        for filename in os.listdir(data_dir):
            if filename.startswith("df_golden_questions"):
                df = pd.read_csv(data_dir+'/'+filename)

                # #sample 20 examples
                # df = df.sample(n=20).reset_index(drop=True)

                style = filename.split('df_golden_questions_')[-1].split('.')[0]
                model.rag_use_questions = False
                model.rag_use_summary = False
                model.rag_on = False
                # generate_answers_from_dataset(model, df.copy(), output_name=f"outputs_no_rag_{style}.csv") # Sem RAG
                model.rag_on = True 
                model.rag_use_text = True  
                generate_answers_from_dataset(model, df, output_name=f"outputs_rag_{style}.csv") # Apenas chunks
                try:
                    model.rag_use_text = False
                    model.rag_use_questions = True
                    # generate_answers_from_dataset(model, df, output_name=f"outputs_rag_{style}_questions.csv") # Apenas Perguntas
                    model.rag_use_questions = False
                    model.rag_use_summary = True
                    # generate_answers_from_dataset(model, df, output_name=f"outputs_rag_{style}_summary.csv") # Apenas Resumos
                except Exception as e:
                    print("Error: ", str(e))
                    traceback.print_exc()
                    print("Error while generating answers: ", filename)