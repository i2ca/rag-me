#A script that emulates RAGAS https://arxiv.org/pdf/2309.15217.pdf
# We evaluate a RAG system by 3 metrics:
# Faithfulness - How much the claims in the answer can be extracted from the context
# Answer relevance - Is the answer relevant to the question?
# Context Relevance - Is the context retrieved relevant to the question?

import pandas as pd
import transformers
import torch
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import random
from dotenv import load_dotenv
import os

load_dotenv()

print("Initializing LLM model...")
model_id = 'meta-llama/Llama-2-13b-chat-hf'
# begin initializing HF items, need auth token for these
hf_auth = os.getenv('HUGGINGFACE_AUTH_TOKEN')

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
) 
""" bnb_config = transformers.BitsAndBytesConfig(
    load_in_8bit=True #Use 15Gb of memory
) """
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    token=hf_auth
)   
# initialize the model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    token=hf_auth
)
model.eval()
#Initialize tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    token=hf_auth
)
#Initialize pipeline
llama_pipeline = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    # we pass model parameters here too
    eos_token_id=tokenizer.eos_token_id,
    temperature=0.2,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=1024,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

def calculate_faithfulness(sentences_csv_path, dataset):
    print("Calculating Faithfulness...")
    generate_sentences(sentences_csv_path, dataset)
    score_sentences(sentences_csv_path, dataset)

def generate_sentences(sentences_csv_path, dataset):
    #Create dataframe with prompts
    df_prompts = pd.DataFrame(columns=['question_i', 'answer_type', 'prompt'])
    rand = random.randrange(0, len(dataset.index)-6)
    for i in range(len(dataset.index)): #range(rand, rand+5): 
        line = dataset.iloc[i]
        question = line['question']
        system_answer = line['answer']
        ungrounded_answer = line['ungrounded_answer']
        poor_answer = line['poor_answer']
        prompt = f"""Given a question and answer, create one or more statements from each sentence in the given answer.
{question}
{system_answer}
Statements: """
        df_prompts = pd.concat([df_prompts, pd.DataFrame({'question_i': [i], 'answer_type':['grounded'], 'prompt': [prompt]})], ignore_index=True)

        prompt = f"""Given a question and answer, create one or more statements from each sentence in the given answer.
{question}
{ungrounded_answer}
Statements: """
        df_prompts = pd.concat([df_prompts, pd.DataFrame({'question_i': [i], 'answer_type':['poor'], 'prompt': [prompt]})], ignore_index=True)

        prompt = f"""Given a question and answer, create one or more statements from each sentence in the given answer.
{question}
{poor_answer}
Statements: """
        df_prompts = pd.concat([df_prompts, pd.DataFrame({'question_i': [i], 'answer_type':['ungrounded'], 'prompt': [prompt]})], ignore_index=True)

    dataset_prompts = Dataset.from_pandas(df_prompts)
    count = 0
    df_prompts['sentences'] = ''
    print("Creating sentences from answers...")
    try:
        #There is a bug when using batch size > 1 !!!
        for out in tqdm(llama_pipeline(KeyDataset(dataset_prompts, 'prompt'), batch_size=1)):
            torch.cuda.empty_cache()
            for seq in out:
                line = df_prompts.iloc[count]
                prompt = line['prompt']
                #print(f'\n\n---------------\n\nPrompt:\n{prompt}')
                response = seq['generated_text'][len(prompt):]
                #print(f"\nResponse:\n{response}")

                #add line to df
                df_prompts.at[count, 'sentences'] = response
                count+=1
        print("Finished creating sentences!")
    except:
        print("Error in creating sentences!!")
    df_prompts.to_csv(sentences_csv_path, index=False)

def score_sentences(sentences_csv_path, dataset):
    def parse_faithfulness_score(response):
        response = response.lower()
        if "final verdict" in response:
            final = response.split('final verdict')[1]
            num_yes = final.count('yes')
            num_no = final.count('no')
            if (num_yes+num_no) > 0:
                return (num_yes/(num_yes+num_no))
            else:
                return float('nan')
        else:
            final = response
            num_yes = final.count(' yes ')+final.count(' yes,')+final.count(' yes.')
            num_no = final.count(' no ')+final.count(' no,')+final.count(' no.')
            if (num_yes+num_no) > 0:
                return (num_yes/(num_yes+num_no))
            else:
                return float('nan')

    df_sentences = pd.read_csv(sentences_csv_path)

    #Create dataframe with prompts
    df_prompts = pd.DataFrame(columns=['question_i', 'answer_type', 'prompt'])
    rand = random.randrange(0, len(df_sentences.index)-6)
    for i in range(len(df_sentences.index)): #range(rand, rand+5): 
        line = df_sentences.iloc[i]
        sentences = line['sentences']
        question_i = line['question_i']
        answer_type = line['answer_type']
        context = dataset.iloc[int(question_i)]['context_v1']

        prompt = f"""Consider the given context and following statements, then determine whether they are supported by the information present in the context. 
Provide a brief explanation for each statement before arriving at the verdict (Yes/No). 
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
"""
        df_prompts = pd.concat([df_prompts, pd.DataFrame({'question_i': [question_i], 'answer_type':[answer_type], 'prompt': [prompt]})], ignore_index=True)
        
    dataset_prompts = Dataset.from_pandas(df_prompts)
    count = 0
    df_prompts['faithfulness_eval_reason'] = ''
    df_prompts['faithfulness_eval_score'] = 0.0
    print("Evaluating statements from context...")
    try:
        #There is a bug when using batch size > 1 !!!
        for out in tqdm(llama_pipeline(KeyDataset(dataset_prompts, 'prompt'), batch_size=1)):
            torch.cuda.empty_cache()
            for seq in out:
                line = df_prompts.iloc[count]
                prompt = line['prompt']
                print(f'\n\n---------------\n\nPrompt:\n{prompt}')
                response = seq['generated_text'][len(prompt):]
                print(f"\nResponse:\n{response}")
                score = parse_faithfulness_score(response)
                print('Score: ', score)

                #add line to df
                df_prompts.at[count, 'faithfulness_eval_reason'] = response
                df_prompts.at[count, 'faithfulness_eval_score'] = score
                count+=1
        print("Finished evaluating sentences!")
    except Exception as e:
        print("Error: ", str(e))
        print("Error in evaluating sentences!!")
    df_prompts.to_csv('evaluation/faithfulness/faithfulness_scores2.csv', index=False)


df = pd.read_parquet('exploracao/wikieval/train-00000-of-00001-385c01e94624e9b7.parquet')
calculate_faithfulness('evaluation/faithfulness/sentences.csv', df)