#A script to create a faiss index with questions about multiple books and pdfs

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
import time
import os
import transformers
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import traceback
from dotenv import load_dotenv
from src.gen_model import init_model

load_dotenv()

default_model_id = 'mistralai/Mistral-7B-Instruct-v0.1'
df_questions_path_dir = 'data'
df_questions_name = 'df_questions.csv'
index_dir = 'data'
index_name = 'questions.index'
df_corpus_path_dir = 'data'
df_corpus_name = 'df_corpus_long.csv'
hf_auth = os.getenv('HUGGINGFACE_AUTH_TOKEN')
embedding_model_dimensions = os.getenv("EMBEDDING_MODEL_DIMENSIONS", 1024)

def create_questions(
        df_corpus_path_dir: str = df_corpus_path_dir, 
        df_corpus_name: str = df_corpus_name, 
        df_questions_path_dir: str = df_questions_path_dir,
        df_questions_name: str = df_questions_name,
        model_pipeline: transformers.pipeline = None,
        verbose: bool = False):
    def parse_question_answer(llm_response):
        try:
            #Verify if response is in the format my_question: <question> my_answer: <answer>
            if 'question:' in llm_response and 'answer:' in llm_response:
                question = llm_response.split('question:')[1].split('answer:')[0].strip()
                answer = llm_response.split('answer:')[1].strip()
                #return text between <>
                if '<' in question and '>' in question:
                    question = question[question.index('<')+1:question.index('>')]
                else:
                    question = float('nan')
                if '<' in answer and '>' in answer:
                    answer = answer[answer.index('<')+1:answer.index('>')]
                else:
                    answer = float('nan')
                return question, answer
            else:
                return float('nan'), float('nan')
        except Exception:
            return float('nan'), float('nan')
        
    if verbose:
        print("Creating questions from ", df_corpus_path_dir+'/'+df_corpus_name)
    df_corpus = pd.read_csv(df_corpus_path_dir+'/'+df_corpus_name)

    if model_pipeline is None:
        model_pipeline, _, _ = init_model(model_id=default_model_id, hf_auth=hf_auth, verbose=True)

    SYSTEM_PROMPT = """


I just gave you a chunk of text about theory. Based on this given text, create one question and one answer about the subject that can be answered by the given text. 
It's necessary that the questions aren't especific to the text, they should be answerable by someone without access to the exact text and they should not cite the given text in any way. 
The answers should be very complete, containing all the important information contained in the base text.
Create the question and answer only in the following format with < and > around the question and answer, without adding anything more before or after, because I want to parse the question and answer from the text later:
question: <Example question?>
answer: <This is an example answer to the example question. Make this answer very complete and accurate to the provided context.>

Your turn:
Please provide the question and answer based on the first text:
question: <"""

    #create empty dataframe with columns question, answer, text, init_line
    df_qa = pd.DataFrame(columns=['question', 'answer', 'text', 'init_line', 'source', 'llm_response'])

    #Create dataframe with prompts
    df_prompts = pd.DataFrame(columns=['prompt', 'chunk', 'init_line'])
    for i in range(len(df_corpus.index)):
        line = df_corpus.iloc[i]
        chunk = str(line['text'])
        init_line = line['init_line']
        source = line['source']
        new_message = 'Based on the following text, create a question and answer about the subject:\n\n' + '"'+chunk+'"'
        prompt = new_message+SYSTEM_PROMPT
        df_prompts = pd.concat([df_prompts, pd.DataFrame({'prompt': [prompt], 'chunk': [chunk], 'init_line': [init_line], 'source': [source]})], ignore_index=True)

    dataset = Dataset.from_pandas(df_prompts)
    count = 0
    if verbose:
        print("Generating Questions and Answers...")
    try:
        pbar = tqdm(total=len(dataset))
        for out in model_pipeline(KeyDataset(dataset, 'prompt'), batch_size=1):
            torch.cuda.empty_cache()
            for seq in out:
                line = df_prompts.iloc[count]
                prompt = line['prompt']
                init_line = line['init_line']
                source = line['source']
                chunk = str(line['chunk'])
                #print(out)
                #print(f'\n\n---------------\n\nPrompt:\n{prompt}')
                #print(f"\nResponse:\n{seq['generated_text'][len(prompt):]}")
                response = seq['generated_text'][len(prompt)-len('question: <'):]
                (question, answer) = parse_question_answer(response)

                # if (question == float('nan') or answer == float('nan')):
                    #print("\n\n---------\nNA Question:")
                    #print(response)
                #print(f"\n\nQuestion: {question}")
                #print(f"\n\nAnswer: {answer}")
                #add line to df_qa
                df_qa = pd.concat([df_qa, pd.DataFrame([[question, answer, chunk, init_line, source, response]], columns=df_qa.columns)], ignore_index=True)
                count+=1
                inc = count - pbar.n
                pbar.update(n=inc)
        if verbose:
            print("Finished creating questions and answers!")
            print('Number of NA: ', (df_qa['question'].isna()).sum())
            print('Number of "Example question?": ', (df_qa['question'] == 'Example question?').sum())
        df_qa = df_qa[(df_qa['question'].isna() == False) & (df_qa['answer'].isna() == False)]  # noqa: E712
        df_qa = df_qa[(df_qa['question'] != 'Example question?')]
        df_qa = df_qa.drop_duplicates(subset=['question'])
        #save df_qa to csv
        df_qa.reset_index(drop=True, inplace=True)
        try:
            df_qa.to_csv(df_questions_path_dir+'/'+df_questions_name, index=False)
            if verbose:
                print(f'Saved dataframe to {df_questions_path_dir+"/"+df_questions_name}')
        except Exception as e:
            print(str(e))
            df_qa.to_csv(df_questions_path_dir+'/'+df_questions_name, index=False, escapechar="\\")
            print(f'Saved dataframe to {df_questions_path_dir+"/"+df_questions_name}')
    except Exception as e:
        print("Error: ", str(e))
        traceback.print_exc()
        print("Error in creating questions and answers!!")





####################################
# Create index based on questions
def create_index(
        encoder_model: SentenceTransformer = None,
        encoder_model_dimensions: int = embedding_model_dimensions,
        df_questions_path_dir: str = df_questions_path_dir,
        df_questions_name: str = df_questions_name,
        index_dir: str = index_dir,
        index_name: str = index_name,
        verbose: bool = False
        ):

    if encoder_model is None:
        raise ValueError("Please provide an encoder model")
    df_questions = pd.read_csv(df_questions_path_dir+'/'+df_questions_name)

    if verbose:
        print("Number of questions: ", len(df_questions))
        print("Creating questions index from {}".format(df_questions_path_dir+'/'+df_questions_name))

    #Creation of index
    t=time.time()
    encoded_data = encoder_model.encode([question.strip() for question in df_questions['question'].tolist()],
                                         show_progress_bar=verbose)
    encoded_data = np.asarray(encoded_data.astype('float32'))
    index_blast_furnace = faiss.IndexIDMap(faiss.IndexFlatIP(encoder_model_dimensions))
    index_blast_furnace.add_with_ids(encoded_data, np.array(range(0, len(df_questions))))
    faiss.write_index(index_blast_furnace, index_dir+'/'+index_name)

    if verbose:
        print('>>>> Created embeddings in Total Time: {}'.format(time.time()-t)) 
        print('Index saved to :' ,index_dir+'/'+index_name)
