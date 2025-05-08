#A script that emulates RAGAS https://arxiv.org/pdf/2309.15217.pdf
# We evaluate a RAG system by 3 metrics:
# Faithfulness - How much the claims in the answer can be extracted from the context
# Answer relevance - Is the answer relevant to the question?
# Context Relevance - Is the context retrieved relevant to the question?

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import transformers
import torch
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import random
import re

def calculate_faithfulness(model_pipeline: transformers.pipeline, result_path, calculate_context=True):
    print("\nCalculating Faithfulness...")
    generate_sentences(model_pipeline, result_path)
    try:
        score_sentences(model_pipeline, result_path)
    except Exception as e:
        print(f'Error: {str(e)}\nCould not evaluate faithfulness to original text.')
    if calculate_context:
        score_sentences(model_pipeline, result_path, 'rag_context', column_name="faithfulness_context")

def generate_sentences(model_pipeline: transformers.pipeline, result_path):
    #Read dataset csv
    dataset = pd.read_csv(result_path)
    #Create dataframe with prompts
    df_prompts = pd.DataFrame(columns=['prompt'])
    rand = random.randrange(0, len(dataset.index)-6)
    for i in range(len(dataset.index)): #range(rand, rand+5): 
        line = dataset.iloc[i]
        question = line['question']
        system_answer = line['system_answer']
        prompt = f"""Given a question and answer, create one or more statements from each sentence in the given answer.
{question}
{system_answer}
Statements: """
        df_prompts = pd.concat([df_prompts, pd.DataFrame({'prompt': [prompt]})], ignore_index=True)
    dataset_prompts = Dataset.from_pandas(df_prompts)
    count = 0
    dataset['sentences'] = ''
    print("Creating sentences from answers...")
    try:
        #There is a bug when using batch size > 1 !!!
        pbar = tqdm(total=len(dataset_prompts))
        for out in model_pipeline(KeyDataset(dataset_prompts, 'prompt'), batch_size=16) :
            torch.cuda.empty_cache()
            for seq in out:
                #print("\n\n------\n", seq)
                line = df_prompts.iloc[count]
                prompt = line['prompt']
                #print(f'\n\n---------------\n\nPrompt:\n{prompt}')
                response = seq['generated_text'][len(prompt):]
                #print(f"\nResponse {count}:\n{response}")

                #add line to df
                dataset.at[count, 'sentences'] = response
                count+=1
                inc = count - pbar.n
                pbar.update(n=inc)
        print("Finished creating sentences!")
        dataset.to_csv(result_path, index=False)
    except Exception as e:
        print("Error: ", str(e))
        print("Error in creating sentences!!")

def score_sentences(model_pipeline: transformers.pipeline, result_path, comparing_text='text', column_name='faithfulness_text'):
    def parse_faithfulness_score(response):
        final = response.lower()
        num_yes = final.count(' yes,')+final.count(' yes.')+final.count(' yes\n')+final.count('\nyes')
        num_no = final.count(' no,')+final.count(' no.')+final.count(' no\n')+final.count('\nno')
        if (num_yes+num_no) > 0:
            return (num_yes/(num_yes+num_no))
        else:
            return float('nan')
    #Read dataset csv
    dataset = pd.read_csv(result_path)

    #Create dataframe with prompts
    df_prompts = pd.DataFrame(columns=['prompt'])
    rand = random.randrange(0, len(dataset.index)-6)
    for i in range(len(dataset.index)): # range(rand, rand+5):
        line = dataset.iloc[i]
        sentences = line['sentences']
        context = line[comparing_text]

        prompt = f"""<s>[INST] Consider the given context and following statements, then determine whether they are supported by the information present in the context. 
Provide a brief explanation for each statement before arriving at the verdict (Yes/No), informing if they are supported or not by the context given. 
Provide a final verdict for each statement in order at the end in the following format:
Final Verdict:
Statement 1: [yes/no]
Statement 2: [yes/no]
...
Statement n: [yes/no] 
Do not deviate from the specified format on the final verdict, write only [yes] or [no] for each statement.
Context: 
{context}

Statements: 
{sentences}

Explanation:
[/INST]
"""
        
        df_prompts = pd.concat([df_prompts, pd.DataFrame({'prompt': [prompt]})], ignore_index=True)
        
    dataset_prompts = Dataset.from_pandas(df_prompts)
    count = 0
    dataset[f'{column_name}_eval_reason'] = ''
    dataset[f'{column_name}_eval_score'] = 0.0
    print("Evaluating statements from context...")
    try:
        pbar = tqdm(total=len(dataset_prompts))
        for out in model_pipeline(KeyDataset(dataset_prompts, 'prompt'), batch_size=2):
            torch.cuda.empty_cache()
            for seq in out:
                line = df_prompts.iloc[count]
                prompt = line['prompt']
                #print(f'\n\n---------------\n\nPrompt:\n{prompt}')
                response = seq['generated_text'][len(prompt):]
                #print(f"\nResponse:\n{response}")
                score = parse_faithfulness_score(response)
                #print('Score: ', score)

                #add line to df
                dataset.at[count, f'{column_name}_eval_reason'] = response
                dataset.at[count, f'{column_name}_eval_score'] = score
                count+=1
                inc = count - pbar.n
                pbar.update(n=inc)
        print("Finished evaluating sentences!")
        dataset.to_csv(result_path, index=False)
        print(f'Faithfulness {column_name}: ', dataset[f'{column_name}_eval_score'].mean())
    except Exception as e:
        print("Error: ", str(e))
        print("Error in evaluating sentences!!")




#### ------ <.> -------- ####

def calculate_answer_relevance(model_pipeline: transformers.pipeline, embedding_model: SentenceTransformer, result_path):
    # The ideia here is to evaluate if the answer is relevant to the question
    # We will create n questions based on the answer
    # Then, we will embed the questions and measure the similarity to the real question
    def parse_answer_relevance_score(response, original_question):
        try:
            # Define a regular expression pattern to match the questions
            pattern = re.compile(r'question_\d+: <(.+?)>', re.DOTALL)
            # Use findall to get all matches
            questions = pattern.findall(response)
            #Calculate embeddings for questions
            embedded_questions = embedding_model.encode(questions)
            original_question = embedding_model.encode([original_question])
            #Calculate similarity
            return cosine_similarity(original_question, embedded_questions).reshape(-1).sum()/4
        except:
            return float('nan')
        
    #Read dataset csv
    dataset = pd.read_csv(result_path)
    #Create dataframe with prompts
    df_prompts = pd.DataFrame(columns=['prompt'])
    for i in range(len(dataset.index)): 
        line = dataset.iloc[i]
        system_answer = line['system_answer']

        prompt = f"""Generate 4 questions based on the following answer. Please make sure that each question can be answered by the answer. 
answer: <{system_answer}>
Now give me 4 questions between < and > so that I can parse them later.
QUESTIONS
question_1: <"""
        df_prompts = pd.concat([df_prompts, pd.DataFrame({'prompt': [prompt]})], ignore_index=True)
    
    dataset_prompts = Dataset.from_pandas(df_prompts)
    count = 0
    dataset['answer_relevance_score'] = 0.0
    print("\nGenerating questions for the answer")
    try:
        pbar = tqdm(total=len(dataset_prompts))
        for out in model_pipeline(KeyDataset(dataset_prompts, 'prompt'), batch_size=1):
            torch.cuda.empty_cache()
            for seq in out:
                line = df_prompts.iloc[count]
                prompt = line['prompt']
                answer = dataset.iloc[count]['system_answer']
                original_question = dataset.iloc[count]['question']
                #print(f'\n\n---------------\n\nPrompt:\n{prompt}')
                response = seq['generated_text'][len(prompt)-len('question_1: <'):]
                #print(f"\n\n-------\nAnswer: {answer}")
                #print(f'\nOriginal Question: {original_question}')
                #print(f"\nResponse:\n{response}")
                score = parse_answer_relevance_score(response, original_question)
                if score == float('nan'):
                    print("\nError: couldn't parse response:\n", response)
                #print('Score: ', score)

                #add line to df
                dataset.at[count, 'answer_relevance_score'] = score
                count+=1
                inc = count - pbar.n
                pbar.update(n=inc)
        print("Finished creating questions!")
        dataset.to_csv(result_path, index=False)
        print('Answer Relevance: ', dataset['answer_relevance_score'].mean())
    except Exception as e:
        print("Error: ", str(e))
        print("ERROR in answer relevance!!")
    

#### ------ <.> -------- ####

def calculate_context_relevance(model_pipeline: transformers.pipeline, result_path):
    # The ideia here is to evaluate if the context is relevant to the question
    # We will extract sentences from the context that helps answering the question
    # The score will be n_extracted_senteces/n_total_sentences
    def parse_context_relevance_score(response, context):
        # Define a regular expression pattern to match the questions
        pattern = re.compile(r'GRADE: <(\w+)>', re.DOTALL)
        # Use findall to get all matches
        try:
            grade = pattern.findall(response)[0]
            grade = grade.lower()
            if grade == 'correct':
                return 1
            elif grade == 'incorrect':
                return 0
        except Exception as e:
            print(str(e))
            return 0
        return 0
        

    #Read dataset csv
    dataset = pd.read_csv(result_path)
    #Create dataframe with prompts
    df_prompts = pd.DataFrame(columns=['prompt'])
    for i in range(len(dataset.index)): 
        line = dataset.iloc[i]
        context = line['rag_context']
        question = line['question']
        answer = line['system_answer']

        prompt = f"""
Given the question: \n
{question}

Here are some documents retrieved in response to the question: \n
{context}

And here is the answer to the question: \n 
{answer}

Criteria: 
    relevance: Are the retrieved documents relevant to the question and do they support the answer?"

Your response should be as follows:

GRADE: (<Correct> or <Incorrect>, depending if the retrieved documents meet the criterion. Write < and > around the grade so that I can parse it later)
(line break)
JUSTIFICATION: (Write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Use one or two sentences maximum. Keep the answer as concise as possible.)

GRADE: <"""
        df_prompts = pd.concat([df_prompts, pd.DataFrame({'prompt': [prompt]})], ignore_index=True)
    
    dataset_prompts = Dataset.from_pandas(df_prompts)
    count = 0
    dataset['context_relevance_score'] = 0.0
    dataset['context_relevance_justification'] = ""
    print("\nExtracting relevant sentences from the context")
    try:
        pbar = tqdm(total=len(dataset_prompts))
        for out in model_pipeline(KeyDataset(dataset_prompts, 'prompt'), batch_size=1):
            torch.cuda.empty_cache()
            for seq in out:
                line = df_prompts.iloc[count]
                prompt = line['prompt']
                context = dataset.iloc[count]['rag_context']
                #print(f'\n\n---------------\n\nPrompt:\n{prompt}')
                response = seq['generated_text'][len(prompt)-len('GRADE: <'):]
                #print(f"\nResponse:\n{response}")
                score = parse_context_relevance_score(response, context)
                #print('Score: ', score)

                #add line to df
                dataset.at[count, 'context_relevance_score'] = score
                dataset.at[count, 'context_relevance_justification'] = response
                count+=1
                inc = count - pbar.n
                pbar.update(n=inc)
        print("Finished extracting sentences!")
        dataset.to_csv(result_path, index=False)
        print('Context Relevance: ', dataset['context_relevance_score'].mean())
    except Exception as e:
        print("Error: ", str(e))
        print("ERROR in  extracting sentences!!")