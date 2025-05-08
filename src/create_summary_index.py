#A script to create a faiss index with summary about multiple books and pdfs

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
import time
import os
import transformers
import traceback
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
from dotenv import load_dotenv
try:
    from src.gen_model import init_model
except:
    from gen_model import init_model

load_dotenv()

default_model_id = 'mistralai/Mistral-7B-Instruct-v0.1'
df_summary_path_dir = 'data'
df_summary_name = 'df_summary.csv'
index_dir = 'data'
index_name = 'summary.index'
df_corpus_path_dir = 'data'
df_corpus_name = 'df_corpus_long.csv'
hf_auth = os.getenv('HUGGINGFACE_AUTH_TOKEN')
embedding_model_dimensions = os.getenv("EMBEDDING_MODEL_DIMENSIONS", 1024)

def create_summaries(
        df_corpus_path_dir: str = df_corpus_path_dir, 
        df_corpus_name: str = df_corpus_name, 
        df_summary_path_dir: str = df_summary_path_dir,
        df_summary_name: str = df_summary_name,
        model_pipeline: transformers.pipeline = None,
        verbose: bool = False
        ):
        
    if verbose:
        print("Creating summaries from ", df_corpus_path_dir+'/'+df_corpus_name)
        
    if model_pipeline is None:
        model_pipeline, _, _ = init_model(model_id=default_model_id, hf_auth=hf_auth, verbose=True)
        
    df_corpus = pd.read_csv(df_corpus_path_dir+'/'+df_corpus_name)

    SYSTEM_PROMPT = """
I will give you a full text from a textbook and you will create a summary of it in plain text, please mantain the main information contained in the original text.
Create the summary containing the tecnical information present in the text, do not describe what the text says or reference the text.
For example, write "Some insects like bees are attracted to flowers..." instead of "The text describes which insects are attracted to flowers...".
Please, write the summary in one paragraph, don't answer in topics.

Given text:

"""
    #create empty dataframe with columns question, answer, text, init_line
    df_summary = pd.DataFrame(columns=['summary', 'text', 'init_line', 'source', 'llm_response'])

    #Create dataframe with prompts
    df_prompts = pd.DataFrame(columns=['prompt', 'chunk', 'init_line'])
    for i in range(len(df_corpus.index)):
        line = df_corpus.iloc[i]
        chunk = str(line['text'])
        init_line = line['init_line']
        source = line['source']
        new_message = '"'+chunk+'"'+'\n\nPlease create a summary of the text above:\nSummary:\n'
        prompt = SYSTEM_PROMPT+new_message
        df_prompts = pd.concat([df_prompts, pd.DataFrame({'prompt': [prompt], 'chunk': [chunk], 'init_line': [init_line], 'source': [source]})], ignore_index=True)

    dataset = Dataset.from_pandas(df_prompts)
    count = 0
    try:
        pbar = tqdm(total=len(dataset))
        for out in model_pipeline(KeyDataset(dataset, 'prompt'), batch_size=1):
            torch.cuda.empty_cache()
            for seq in out:
                prompt = df_prompts.iloc[count]['prompt']
                init_line = df_prompts.iloc[count]['init_line']
                source = df_prompts.iloc[count]['source']
                chunk = str(df_prompts.iloc[count]['chunk'])
                response = seq['generated_text'][len(prompt):]

                #print(f"\n\nSummary: {response}")
                #add line to df_summary
                df_summary = pd.concat([df_summary, pd.DataFrame([[response, chunk, init_line, source, response]], columns=df_summary.columns)], ignore_index=True)
                count+=1
                inc = count - pbar.n
                pbar.update(n=inc)
        if verbose:
            print("Finished creating summaries!")
            print('Number of NA: ', (df_summary['summary'].isna()).sum())
        df_summary = df_summary[(df_summary['summary'].isna() == False)]  # noqa: E712
        #save df_summary to csv
        df_summary.reset_index(drop=True, inplace=True)
        df_summary.to_csv(df_summary_path_dir+'/'+df_summary_name, index=False, escapechar="\\")
        if verbose:
            print("Saved summaries to: ", df_summary_path_dir+'/'+df_summary_name)
            print("Number of summaries: ", len(df_summary))
    except Exception as e:
        traceback.print_exc()
        print("ERROR: ", str(e))
        print("Error in creating summaries!!")

####################################
# Create index based on questions
def create_index(
    encoder_model: SentenceTransformer = None,
    encoder_model_dimensions: int = embedding_model_dimensions,
    df_summary_path_dir: str = df_summary_path_dir,
    df_summary_name: str = df_summary_name,
    index_dir: str = index_dir,
    index_name: str = index_name,
    verbose: bool = False):

    if encoder_model is None:
        raise ValueError("Encoder model cannot be None")
    df_summary = pd.read_csv(df_summary_path_dir+'/'+df_summary_name)

    if verbose:
        print("Creating summary index from ", df_summary_path_dir+'/'+df_summary_name)
        print("Number of summaries: ", len(df_summary))

    #Creation of index
    t=time.time()
    encoded_data = encoder_model.encode([summary.strip() for summary in df_summary['summary'].tolist()], 
                                        show_progress_bar=verbose)
    encoded_data = np.asarray(encoded_data.astype('float32'))
    index_blast_furnace = faiss.IndexIDMap(faiss.IndexFlatIP(encoder_model_dimensions))
    index_blast_furnace.add_with_ids(encoded_data, np.array(range(0, len(df_summary))))
    faiss.write_index(index_blast_furnace, index_dir+'/'+index_name)

    if verbose:
        print('>>>> Created embeddings in Total Time: {}'.format(time.time()-t)) 
        print(f'Index saved to: ', index_dir+'/'+index_name)
