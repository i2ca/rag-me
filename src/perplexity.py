#https://huggingface.co/docs/transformers/v4.15.0/perplexity

import sys
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from transformers import MistralForCausalLM
try:
    from src.lcad_rag import LcadRag
except Exception:
    from lcad_rag import LcadRag

# results_dir = 'results/wikipedia_ad_rewrite'
answer_column = 'correct_answer'

def calculate_perplexity(rag: LcadRag, results_path: str):
    print("Calculating perplexity for ", results_path)
    df_results = pd.read_csv(results_path)
    df_results['perplexity'] = float(0)
    model: MistralForCausalLM = rag.model
    tokenizer = rag.tokenizer
    model.eos_token_id=tokenizer.eos_token_id
    model.pad_token_id=tokenizer.eos_token_id
    #Get questions
    questions = df_results['question']
    #Get contexts
    contexts = df_results['rag_context']
    #Get correct answers
    correct_answers = df_results[answer_column]
    #Concat prompts
    if contexts[0] == "":
        prompts = [(rag.prompt+f"{question} [/INST]") for question in questions]
    else:
        prompts = []
        for i in range(len(questions)):
            context = contexts[i]
            question = questions[i]
            prompts += [(rag.rag_prompt.format(text_books = context)+f"{question}[/INST]")]
    for i in tqdm(range(len(prompts))):
        answer = str(correct_answers[i]).strip()
        prompt = prompts[i]
        prompt_len = len(tokenizer(prompt, return_tensors='pt').to('cuda').input_ids[0])
        encodings = tokenizer(prompt+answer, return_tensors='pt')
        input_ids = encodings.input_ids.to('cuda')[:,:]
        target_ids = input_ids.clone()
        target_ids[:,:prompt_len] = -100
        
        with torch.no_grad():
            #print(torch.cuda.memory_reserved())
            loss = model.forward(input_ids=input_ids, labels=target_ids).loss
            #print(torch.cuda.memory_reserved())
            #print('loss: ', float(loss))
            ppl = float(torch.exp(loss))
            df_results.at[i, 'perplexity'] = float(ppl)
            # if ppl > 10000:
            #     print("\n\n\n\n-----------\n\nPerplexity: ", ppl)
            #     print("Loss: ", loss)
            #     print("Prompt: ", tokenizer.batch_decode(input_ids[:,:prompt_len]))
            #     print("\nAnswer: ", tokenizer.batch_decode(input_ids[:,prompt_len:][0], skip_special_tokens=False))
    print("Finished evaluating perplexity! ", results_path)
    df_results.to_csv(results_path, index=False)
    print('Mean Ppl: ', df_results['perplexity'].mean())

def calculate_perplexity_accuracy(rag: LcadRag, results_path: str):
    print("Calculating perplexity accuracy for ", results_path)
    df_results = pd.read_csv(results_path)
    df_results['perplexity_accuracy'] = int(0)
    df_results['perplexity_accuracy_choice'] = int(0)
    model: MistralForCausalLM = rag.model
    tokenizer = rag.tokenizer
    model.eos_token_id=tokenizer.eos_token_id
    model.pad_token_id=tokenizer.eos_token_id
    #Get questions
    questions = df_results['question']
    #Get contexts
    contexts = df_results['rag_context']
    #Get correct answers
    correct_answers = df_results[answer_column]
    wrong_answers1 = df_results['wrong_answer1']
    wrong_answers2 = df_results['wrong_answer2']
    wrong_answers3 = df_results['wrong_answer3']
    #Concat prompts
    if contexts[0] == "":
        prompts = [(rag.prompt+f"{question} [/INST]") for question in questions]
    else:
        prompts = []
        for i in range(len(questions)):
            context = contexts[i]
            question = questions[i]
            formatted_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": (rag.rag_prompt.format(text_books = context)+f"{question}")},
                ], 
                tokenize=False, 
                add_generation_prompt=True
            )
            prompts += [formatted_prompt]
    for i in tqdm(range(len(prompts))):
        answers = (str(correct_answers[i]), str(wrong_answers1[i]), str(wrong_answers2[i]), str(wrong_answers3[i]))
        prompt = prompts[i]+"\n"
        prompt_len = len(tokenizer(prompt, return_tensors='pt').to('cuda').input_ids[0])
        perplexities_answers = [0, 0, 0, 0]
        # print(f"\n------\nPrompt: {prompt}")
        for i_answer in range(len(answers)):
            prompt_answer = prompt+answers[i_answer]
            encodings = tokenizer(prompt_answer, return_tensors='pt')
            input_ids = encodings.input_ids.to('cuda')
            target_ids = input_ids.clone()
            target_ids[:,:prompt_len] = -100
            
            # print(f"\n--Answer: {answers[i_answer]}")
            # print(f"Target Ids: {target_ids}")
            # print(f"Answer: {i_answer} - Correct Answer: {0}")
            with torch.no_grad():
                loss = model.forward(input_ids=input_ids, labels=target_ids).loss
                # print('loss: ', float(loss))
                ppl = float(torch.exp(loss))
                # print('ppl: ', ppl)
                perplexities_answers[i_answer] = ppl
        choice = np.argmin(perplexities_answers)
        df_results.at[i, 'perplexity_accuracy_choice'] = choice
        if choice == 0:
            df_results.at[i, 'perplexity_accuracy'] = 1
        else:
            df_results.at[i, 'perplexity_accuracy'] = 0
    print("Finished evaluating perplexity_accuracy! ", results_path)
    df_results.to_csv(results_path, index=False)
    print('Mean Ppl Accuracy: ', df_results['perplexity_accuracy'].mean())

dict_choices = {0: "A", 1: "B", 2: "C", 3: "D"}
inverse_dict_choices = {"A": 0, "B": 1, "C": 2, "D": 3}

def format_choices(correct_answer, wrong_answer1, wrong_answer2, wrong_answer3, choices_order):
    choices = [correct_answer, wrong_answer1, wrong_answer2, wrong_answer3]
    # Sort according to choices_order
    if len(choices_order) != 4:
        raise ValueError("choices_order must be of length 4")
    choices = [choices[int(idx)] for idx in choices_order]
    correct_answer_index = choices.index(correct_answer)
    correct_answer = dict_choices[correct_answer_index]
    #Format choices in a string:
    #- A) choice 1
    #- B) choice 2
    #- C) choice 3
    #- D) choice 4
    string_choices = ""
    for i in range(len(choices)):
        string_choices += f"- {dict_choices[i]}) {choices[i]}\n"

    return string_choices, correct_answer, choices

def calculate_perplexity_accuracy_options(rag: LcadRag, results_path: str):
    print("Calculating perplexity accuracy for ", results_path)
    df_results = pd.read_csv(results_path)
    df_results['perplexity_accuracy_options'] = int(0)
    df_results['perplexity_accuracy_options_choice'] = int(0)
    model: MistralForCausalLM = rag.model
    tokenizer = rag.tokenizer
    model.eos_token_id=tokenizer.eos_token_id
    model.pad_token_id=tokenizer.eos_token_id
    #Get questions
    questions = df_results['question']
    #Get contexts
    contexts = df_results['rag_context']
    #Get correct answers
    correct_answers = df_results[answer_column]
    wrong_answers1 = df_results['wrong_answer1']
    wrong_answers2 = df_results['wrong_answer2']
    wrong_answers3 = df_results['wrong_answer3']
    choices_orders = df_results['choices_order']
    
    # CREATE PROMPTS
    if choices_orders is None:
        raise ValueError("choices_order cannot be None. Run script exploracao/add_order_to_qa_dataset.py")
    #Concat prompts
    prompts = pd.DataFrame(columns=['prompt', 'correct_answer', 'choices'])
    for i in range(len(questions)):
        context = contexts[i]
        question = questions[i]
        choices_order = list(eval(choices_orders[i]))

        choices, correct_answer, choices_list = format_choices(
            correct_answers[i], 
            wrong_answers1[i], 
            wrong_answers2[i], 
            wrong_answers3[i], 
            choices_order=choices_order
        )
        try:
            if np.isnan(context):
                context_prompt = ""
        except:
            pass
        if context == "" or context == float('nan') or context == 'nan':
            context_prompt = ""
        else:
            context_prompt = f"Based on the following context, answer the last question:\nContext: {context}\n"
        prompt = \
f"""{context_prompt}

The following are multiple choice questions (with answers).

Question: What is the capital city of France?

Choices:
- A) London
- B) Paris
- C) Rome
- D) Berlin

Correct Answer: B)

Question: Who wrote the novel "Pride and Prejudice"?

Choices:
- A) Emily Brontë
- B) Charlotte Brontë
- C) Jane Austen
- D) Virginia Woolf

Correct Answer: A)

Question: What is the biggest river in the world?

Choices:
- A) Nile
- B) Amazon
- C) Yangtze
- D) Yangtze

Correct Answer: A)

Question: What is the smallest country in the world?

Choices:
- A) Suriname
- B) Monaco
- C) Luxemburg
- D) Vatican City

Correct Answer: D)

Question: What is the highest mountain in the world?

Choices:
- A) Pico da Bandeira
- B) Mount Kilimanjaro
- C) Mount Everest
- D) Mount Fuji

Correct Answer: C)

Answer the following question based on the context above in the same format:
Question: {question}

Choices: 
{choices}
Correct Answer:"""
        
        formatted_prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
            ], 
            tokenize=False, 
            add_generation_prompt=True
        )

        prompts = pd.concat([prompts, 
                                 pd.DataFrame({'prompt': [formatted_prompt], 'correct_answer': [correct_answer], 'choices': [choices_list]})], ignore_index=True)
        



    for i in tqdm(range(len(prompts.index))):
        answers = list(inverse_dict_choices.keys())
        prompt_row = prompts.iloc[i]
        prompt = str(prompt_row['prompt'])
        correct_answer = str(prompt_row['correct_answer'])
        prompt_len = len(tokenizer(prompt+"\n", return_tensors='pt').to('cuda').input_ids[0])
        perplexities_answers = [0, 0, 0, 0]
        # print("\n-----")
        for i_answer in range(len(answers)):
            encodings = tokenizer(prompt+"\n"+str(answers[i_answer]), return_tensors='pt')
            input_ids = encodings.input_ids.to('cuda')
            target_ids = input_ids.clone()
            target_ids[:,:prompt_len] = -100
            # print("\n---------->\nPrompt: ", prompt+str(answers[i_answer]))
            # print("\n------\nTarget Ids: ", target_ids)
            # print(f"Answer: {answers[i_answer]} - Correct Answer: {correct_answer}")
            with torch.no_grad():
                loss = model.forward(input_ids=input_ids, labels=target_ids).loss
                # print('loss: ', float(loss))
                ppl = float(torch.exp(loss))
                # print('ppl: ', ppl)
                perplexities_answers[i_answer] = ppl
        choice = np.argmin(perplexities_answers)
        df_results.at[i, 'perplexity_accuracy_options_choice'] = choice
        if dict_choices[choice] == correct_answer:
            df_results.at[i, 'perplexity_accuracy_options'] = 1
        else:
            df_results.at[i, 'perplexity_accuracy_options'] = 0
    print("Finished evaluating perplexity_accuracy_options! ", results_path)
    df_results.to_csv(results_path, index=False)
    print('Mean Ppl Accuracy: ', df_results['perplexity_accuracy_options'].mean())

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    # Change this to inform where the RAG data is and where the results should be saved inside the dir evaluation/
    data_dir = os.getenv('DATA_DIR', 'data/default')
    results_dir = os.getenv('RESULTS_DIR', 'results/default')
    model_id = os.getenv("GEN_MODEL", 'mistralai/Mistral-7B-Instruct-v0.3')
    model = LcadRag(model_id=model_id, rag_on=True, rag_use_text=True, rag_use_questions=False, \
                            rag_use_summary=False, rag_query_expansion=False, data_dir=data_dir)
    #Get every filename inside results_dir
    filenames = os.listdir(results_dir)
    for filename in filenames:
    #     calculate_perplexity(lcad_rag, results_dir+'/'+filename)
        # calculate_perplexity_accuracy_options(model, results_dir+'/'+filename)
        calculate_perplexity_accuracy(model, results_dir+'/'+filename)
