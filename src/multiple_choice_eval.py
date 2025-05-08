# Baseado em https://huggingface.co/blog/open-llm-leaderboard-mmlu

import gc
import json
import random
import pandas as pd
import numpy as np
import os
import traceback
from tqdm import tqdm
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
import torch

try:
    from src.gen_model import generate_text, init_model
    from src.lcad_rag import LcadRag
except:
    from lcad_rag import LcadRag
    from gen_model import generate_text, init_model

results_dir = "results/cvpr-papers-llama3"

answer_column = 'correct_answer' #'answer'

dict_choices = {0: "A", 1: "B", 2: "C", 3: "D"}
inverse_dict_choices = {"A": 0, "B": 1, "C": 2, "D": 3}
evaluator_model_id = "meta-llama/Llama-3.1-70B-Instruct"
# evaluator_model_id = "gpt-4o"
# evaluator_model_id = "Qwen/Qwen2.5-7B-Instruct"
# evaluator_model_id = "meta-llama/Llama-3.3-70B-Instruct"

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

def calculate_multiple_choice_accuracy(rag: LcadRag, csv_path: str):
    print("Calculating multiple choice accuracy for ", csv_path)
    df_results = pd.read_csv(csv_path)
    df_results['multiple_choice_correct_answer'] = ""
    df_results['multiple_choice_answer'] = ""
    df_results['multiple_choice_accuracy'] = float('nan')
    model_pipeline = rag.model_pipeline
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

        prompts = pd.concat([prompts, 
                                 pd.DataFrame({'prompt': [prompt], 'correct_answer': [correct_answer], 'choices': [choices_list]})], ignore_index=True)
    
    dataset_prompts = Dataset.from_pandas(prompts)
    try:
        count = 0
        pbar = tqdm(total=len(dataset_prompts))
        for out in model_pipeline(KeyDataset(dataset_prompts, 'prompt'), batch_size=1, max_new_tokens=15):
            torch.cuda.empty_cache()
            for seq in out:
                line = prompts.iloc[count]
                prompt = line['prompt']
                correct_answer = line['correct_answer']
                choices = line['choices']
                response = seq['generated_text'][len(prompt):]
                # print(f'\n\n---------------\n\nPrompt:\n{seq["generated_text"]}')
                answer = str(response).strip().upper()
                if answer != '':
                    answer = answer[0]
                # print("Answer: ", answer)
                # print("Correct Answer: ", correct_answer)
                
                df_results.at[count, 'multiple_choice_correct_answer'] = correct_answer
                if answer not in inverse_dict_choices:
                    df_results.at[count, 'multiple_choice_answer'] = \
                                    str(answer)
                else:
                    df_results.at[count, 'multiple_choice_answer'] = \
                                    str(str(answer) + ' - ' + str(choices[inverse_dict_choices[answer]]))
                df_results.at[count, 'multiple_choice_accuracy'] = 1 \
                                    if answer == str(correct_answer).strip()[0] \
                                    else 0
                
                pbar.set_description(f"Accuracy: {df_results['multiple_choice_accuracy'].mean():.4f}")
                count+=1
                inc = count - pbar.n
                pbar.update(n=inc)
        print("Finished evaluating multiple choice questions!")
        df_results.to_csv(csv_path, index=False)
        print('Multiple Choice Accuracy: ', df_results['multiple_choice_accuracy'].mean())
    except Exception as e:
        traceback.print_exc()
        print("Error: ", str(e))
        print("Error in evaluating multiple choice questions!!")







def prompt_mca_v2(reference_text: str, question: str, system_answer: str, options: str):
    prompt = (
        "Given a base truth text, a question, a candidate answer, "
        "and a set of answer options, tell me\n"
        "which answer option is the closest to the candidate answer." 
        "I don't want to know if the answers are "
        "correct or not, "
        "only which answer is the closest to the candidate answer, "
        "which can be correct or incorrect.\n\n"
        "Base truth:\n"
        f"{reference_text}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Candidate Answer:\n"
        f"{system_answer}\n\n"
        "Answer Options:\n"
        f"{options}\n\n"
        "Now give me the answer option that has the most similar meaning to the candidate answer, "
        "independently of it correctly or incorrectly answering the question.\n"
        "Answer only with the letter of the option (A, B, C or D) and nothing more. "
        "Don't try to explain your answer.\n"
        "If none of the answer options is similar to the candidate answer, answer 'none'.\n"
    )
    return prompt

def prompt_mca_v2_nocontext(reference_text: str, question: str, system_answer: str, options: str):
    prompt = (
        "Given a question, a candidate answer, and a set of answer options, tell me\n"
        "which answer option is closest to the candidate answer. I don't want to know if the answers are correct or not, "
        "only which answer is closest to the candidate, which can be a good or a bad answer.\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Candidate Answer:\n"
        f"{system_answer}\n\n"
        "Answer Options:\n"
        f"{options}\n\n"
        "Now give me the answer option that has the most similar meaning to the candidate answer, "
        "independently of it being correct or incorrect.\n"
        "Answer only with the letter of the option (A, B, C or D) and nothing more. Don't try to explain your answer. "
        "If none of the answer options is similar to the candidate answer, answer 'none'.\n"
    )
    return prompt

def prompt_mca_v2_cot_reason(reference_text: str, question: str, system_answer: str, options: str):
    prompt = (
        "Given a base truth text, a question, a candidate answer, "
        "and a set of answer options, tell me\n"
        "which answer option is the closest to the candidate answer." 
        "I don't want to know if the answers are "
        "correct or not, "
        "only which answer is the closest to the candidate answer, "
        "which can be correct or incorrect.\n\n"
        "Base truth:\n"
        f"{reference_text}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Candidate Answer:\n"
        f"{system_answer}\n\n"
        "Answer Options:\n"
        f"{options}\n\n"
        "Now compare the candidate answer to each of the answer options. DO NOT consider the Base Truth text in the comparison. "
        "Think step by step and explain your chain of thought. "
        "If none of the answer options is similar enough to the candidate answer, "
        "conclude that none of the answer options is similar.\n"
    )
    return prompt

def prompt_mca_v2_cot_reason_nocontext(question: str, system_answer: str, options: str):
    prompt = (
        "Given a question, a candidate answer, "
        "and a set of answer options, tell me\n"
        "which answer option is the closest to the candidate answer." 
        "I don't want to know if the answers are "
        "correct or not, "
        "only which answer is the closest to the candidate answer, "
        "which can be correct or incorrect.\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Candidate Answer:\n"
        f"{system_answer}\n\n"
        "Answer Options:\n"
        f"{options}\n\n"
        "Now compare the candidate answer to each of the answer options."
        "Think step by step and explain your chain of thought. "
        "If none of the answer options is similar enough to the candidate answer, "
        "conclude that none of the answer options is similar.\n"
    )
    return prompt

def prompt_mca_v2_cot_choose(question: str, system_answer: str, options: str, reasoning: str):
    prompt = (
        "Given a question, a candidate answer, "
        "and a set of answer options, tell me\n"
        "which answer option is the closest to the candidate answer." 
        "I don't want to know if the answers are "
        "correct or not, "
        "only which answer is the closest to the candidate answer, "
        "which can be correct or incorrect.\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Candidate Answer:\n"
        f"{system_answer}\n\n"
        "Answer Options:\n"
        f"{options}\n\n"
        "Now based on the following reasoning and comparison, choose the answer option that is the closest to the candidate answer.\n"
        "Reasoning:\n"
        f"{reasoning}\n\n"
        "Answer only with the letter of the closest option (A, B, C or D) and nothing more. Don't try to explain your answer. "
        "If none of the answer options is similar to the candidate answer, answer 'none'.\n"
    )
    return prompt




def calculate_mca_v2(csv_path: str, hf_auth: str):
    print("Calculating MCA V2 for ", csv_path)
    df_results = pd.read_csv(csv_path)
    df_results['mca_v2_answer'] = ""
    df_results['mca_v2'] = float('nan')

    print("Initializing evaluator model: ")
    try:
        evaluator_pipeline, _, _ = init_model(evaluator_model_id, hf_auth, verbose=True)
    except Exception as e:
        print("Error initializing evaluator model: ", str(e))
        print("Check GPU memory available and try again")
        return

    # Iterate over the rows in the DataFrame
    for index, row in tqdm(df_results.iterrows(), total=len(df_results)):
        # Get the question and correct answer
        reference_text = row['text']
        question = row['question']
        system_answer = row['system_answer']
        choices_order = list(eval(row['choices_order']))
        options = format_choices(row['correct_answer'], row['wrong_answer1'], row['wrong_answer2'], row['wrong_answer3'], choices_order)

        prompt = prompt_mca_v2(reference_text, question, system_answer, options)
        prompt_no_context = prompt_mca_v2_nocontext(reference_text, question, system_answer, options)
        gen_option = generate_text(evaluator_pipeline, prompt, max_new_tokens=4)
        gen_option_no_context = generate_text(evaluator_pipeline, prompt_no_context, max_new_tokens=4)
        # print(">> Generated Option: ", gen_option)
        option = gen_option.strip().upper().replace("'", '').replace('"', '').replace("-", '')[0]
        option_no_context = gen_option_no_context.strip().upper().replace("'", '').replace('"', '').replace("-", '')[0]

        df_results.at[index, 'mca_v2_answer'] = option
        df_results.at[index, 'mca_v2_answer_no_context'] = option_no_context
        correct_option = 0
        correct_option_no_context = 0
        if option in inverse_dict_choices.keys():
            if choices_order[inverse_dict_choices[option]] == 0:
                correct_option = 1
        else:
            # print("Error - Option not found: ", gen_option)
            pass
        df_results.at[index, 'mca_v2'] = correct_option

        if option_no_context in inverse_dict_choices.keys():
            if choices_order[inverse_dict_choices[option_no_context]] == 0:
                correct_option_no_context = 1
        else:
            # print("Error - Option not found - No Context: ", gen_option_no_context)
            pass
        df_results.at[index, 'mca_v2_no_context'] = correct_option_no_context
    print("Finished evaluating mca_v2!")
    df_results.to_csv(csv_path, index=False)
    print('mca_v2: ', df_results['mca_v2'].mean())

    del(evaluator_pipeline)
    gc.collect()
    torch.cuda.empty_cache()


def calculate_mca_v2_chain_of_thought(csv_path: str, hf_auth: str):
    print("Calculating MCA V2 CoT for ", csv_path)
    df_results = pd.read_csv(csv_path)
    try:
        # check if df_results has column mca_v2_cot_no_context
        if df_results['mca_v2_cot_no_context'].mean() > 0:
            return
    except KeyError:
        pass
    # df_results['mca_v2_cot_answer'] = ""
    # df_results['mca_v2_cot_reason'] = ""
    # df_results['mca_v2_cot'] = float('nan')
    df_results['mca_v2_cot_answer_no_context'] = ""
    df_results['mca_v2_cot_reason_no_context'] = ""
    df_results['mca_v2_cot_no_context'] = float('nan')

    print("Initializing evaluator model: ")
    try:
        evaluator_pipeline, _, _ = init_model(evaluator_model_id, hf_auth, verbose=True)
    except Exception as e:
        # stack trace
        print(traceback.format_exc())
        print("Error initializing evaluator model: ", str(e))
        print("Check GPU memory available and try again")
        return

    # Iterate over the rows in the DataFrame
    for index, row in tqdm(df_results.iterrows(), total=len(df_results)):
        # Get the question and correct answer
        reference_text = row['text']
        question = row['question']
        system_answer = row['system_answer']
        choices_order = list(eval(row['choices_order']))
        options = format_choices(row['correct_answer'], row['wrong_answer1'], row['wrong_answer2'], row['wrong_answer3'], choices_order)

        # prompt = prompt_mca_v2_cot_reason(reference_text, question, system_answer, options)
        # gen_reasoning = generate_text(evaluator_pipeline, prompt, max_new_tokens=2048)
        # prompt = prompt_mca_v2_cot_choose(question, system_answer, options, gen_reasoning)
        # gen_option = generate_text(evaluator_pipeline, prompt, max_new_tokens=4)
        # # print(">> Generated Option: ", gen_option)
        # option = gen_option.strip().upper().replace("'", '').replace('"', '').replace("-", '')[0]

        # df_results.at[index, 'mca_v2_cot_answer'] = option
        # df_results.at[index, 'mca_v2_cot_reason'] = gen_reasoning
        # correct_option = 0
        # if option in inverse_dict_choices.keys():
        #     if choices_order[inverse_dict_choices[option]] == 0:
        #         correct_option = 1
        # else:
        #     # print("Error - Option not found: ", gen_option)
        #     pass
        # df_results.at[index, 'mca_v2_cot'] = correct_option

        # No context
        prompt = prompt_mca_v2_cot_reason_nocontext(question, system_answer, options)
        gen_reasoning = generate_text(evaluator_pipeline, prompt, max_new_tokens=2048)
        prompt = prompt_mca_v2_cot_choose(question, system_answer, options, gen_reasoning)
        gen_option = generate_text(evaluator_pipeline, prompt, max_new_tokens=4)
        option_no_context = gen_option.strip().upper().replace("'", '').replace('"', '').replace("-", '')[0]

        df_results.at[index, 'mca_v2_cot_answer_no_context'] = option_no_context
        df_results.at[index, 'mca_v2_cot_reason_no_context'] = gen_reasoning
        correct_option = 0
        if option_no_context in inverse_dict_choices.keys():
            if choices_order[inverse_dict_choices[option_no_context]] == 0:
                correct_option = 1
        else:
            # print("Error - Option not found: ", gen_option)
            pass
        df_results.at[index, 'mca_v2_cot_no_context'] = correct_option

#         print(f"\n\n\n\n---------\n\nQuestion: {question}\n\n\
# Reference Answer: {system_answer}\n\n\
# Reasoning: {gen_reasoning}\n\n\
# Option: {gen_option}\n\
# Correct Option: {correct_option}\n\
# Correct Reference: {row['human_eval']}\n\n")

    print("Finished evaluating mca_v2_cot!")
    df_results.to_csv(csv_path, index=False)
    print('mca_v2_cot: ', df_results['mca_v2_cot_no_context'].mean())

    del(evaluator_pipeline)
    gc.collect()
    torch.cuda.empty_cache()





def _prompt_da_cot_reason(reference_text, question, system_answer, max_score=5):
    prompt = (
        "You are a college professor.\n"
        "Given a base truth text, a question and a student answer give me "
        f"a score from 1 to {max_score} on the technical accuracy of the answer.\n\n"
        "Base truth:\n"
        f"{reference_text}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Candidate Answer:\n"
        f"{system_answer}\n\n"
        f"Now consider the Base Truth and evaluate the technical accuracy of the answer, giving a score from 1 to {max_score} "
        f"- being 1 a completely innacurate answer and {max_score} a completely accurate answer.\n"
        "Think step by step and explain your chain of thought. "
    )
    return prompt


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


def calculate_direct_assessment(csv_path: str, hf_auth: str):
    print("Calculating Direct Assessment CoT for ", csv_path)
    df_results = pd.read_csv(csv_path)
    try:
        # check if df_results has column mca_v2_cot_no_context
        if df_results['da_cot_10'].mean() > 0:
            print("Direct Assessment CoT already calculated")
            return
        else:
            if len(str(df_results['da_cot_10_reason'].iloc[0])) <= 0:
                df_results['da_cot_10_reason'] = ""
                df_results['da_cot_10_score'] = float('nan')
            else:
                print("Using cached Direct Assessment CoT")
    except KeyError:
        df_results['da_cot_10_reason'] = ""
        df_results['da_cot_10_score'] = float('nan')
        df_results['da_cot_10'] = float('nan')

    print("Initializing evaluator model: ")
    try:
        evaluator_pipeline, _, _ = init_model(evaluator_model_id, hf_auth, verbose=True)
    except Exception as e:
        print(traceback.format_exc())
        print("Error initializing evaluator model: ", str(e))
        print("Check GPU memory available and try again")
        return

    # Iterate over the rows in the DataFrame
    for index, row in tqdm(df_results.iterrows(), total=len(df_results)):
        # Get the question and correct answer
        reference_text = row['text']
        question = row['question']
        system_answer = row['system_answer']

        ### Max score 10
        if len(str(row['da_cot_10_reason'])) <= 0:
            prompt = _prompt_da_cot_reason(reference_text, question, system_answer, max_score=10)
            gen_reasoning = generate_text(evaluator_pipeline, prompt, max_new_tokens=2048)
        else:
            gen_reasoning = row['da_cot_10_reason']
        prompt = _prompt_da_cot_choose(question, system_answer, gen_reasoning, max_score=10)
        gen_option = generate_text(evaluator_pipeline, prompt, max_new_tokens=5)
        # print(">> Generated Option: ", gen_option)
        try:
            try:
                score = int(gen_option.strip().upper().replace("'", '').replace('"', '').replace("-", '')[:2])
            except Exception as e:
                score = int(gen_option.strip().upper().replace("'", '').replace('"', '').replace("-", '')[0])
        except Exception as e:
            score = -1

        df_results.at[index, 'da_cot_10_score'] = score
        df_results.at[index, 'da_cot_10_reason'] = gen_reasoning
        correct_option = 0
        if score > 6:
            correct_option = 1
        df_results.at[index, 'da_cot_10'] = correct_option

    print("Finished evaluating da_cot!")
    df_results.to_csv(csv_path, index=False)
    print('da_cot: ', df_results['da_cot_10'].mean())

    del(evaluator_pipeline)
    gc.collect()
    torch.cuda.empty_cache()



def _prompt_da_cot_ex_reason(reference_text, question, system_answer, correct_answer: str, wrong_answers: list[str], max_score=5):
    prompt = (
        "You are a college professor.\n"
        "Given a base truth text, a question and a student answer, give me "
        f"a score from 1 to {max_score} on the technical accuracy of the answer.\n\n"
        "Base truth:\n"
        f"{reference_text}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Examples:\n"
        "Answer: {correct_answer}\n"
        "The technical accuracy of this answer is: {max_score}\n"
        "Answer: {wrong_answers[0]}\n"
        "The technical accuracy of this answer is: 1\n"
        "Answer: {wrong_answers[1]}\n"
        "The technical accuracy of this answer is: 1\n"
        "Answer: {wrong_answers[2]}\n"
        "The technical accuracy of this answer is: 1\n\n"
        "Candidate Answer:\n"
        f"{system_answer}\n\n"        
        f"Now consider the Base Truth and the examples and evaluate the technical accuracy of the candidate answer, giving a score from 1 to {max_score} "
        f"- being 1 a completely innacurate answer and {max_score} a completely accurate answer.\n"
        "Think step by step and explain your chain of thought. "
    )
    return prompt


def calculate_direct_assessment_examples(csv_path: str, hf_auth: str):
    print("Calculating Direct Assessment CoT for ", csv_path)
    df_results = pd.read_csv(csv_path)
    df_results['da_cot_ex_reason'] = ""
    df_results['da_cot_ex_score'] = float('nan')
    df_results['da_cot_ex'] = float('nan')

    print("Initializing evaluator model: ")
    try:
        evaluator_pipeline, _, _ = init_model(evaluator_model_id, hf_auth, verbose=True)
    except Exception as e:
        print("Error initializing evaluator model: ", str(e))
        print("Check GPU memory available and try again")
        return

    # Iterate over the rows in the DataFrame
    for index, row in tqdm(df_results.iterrows(), total=len(df_results)):
        # Get the question and correct answer
        reference_text = row['text']
        question = row['question']
        system_answer = row['system_answer']
        choices = [row['correct_answer'], row['wrong_answer1'], row['wrong_answer2'], row['wrong_answer3']]

        ### Max score 10
        prompt = _prompt_da_cot_ex_reason(reference_text, question, system_answer, correct_answer=choices[0], wrong_answers=choices[1:], max_score=10)
        gen_reasoning = generate_text(evaluator_pipeline, prompt, max_new_tokens=2048)
        prompt = _prompt_da_cot_choose(question, system_answer, gen_reasoning, max_score=10)
        gen_option = generate_text(evaluator_pipeline, prompt, max_new_tokens=4)
        # print(">> Generated Option: ", gen_option)
        try:
            score = int(gen_option.strip().upper().replace("'", '').replace('"', '').replace("-", '')[0])
        except Exception as e:
            score = -1

        df_results.at[index, 'da_cot_ex_score'] = score
        df_results.at[index, 'da_cot_ex_reason'] = gen_reasoning
        correct_option = 0
        if score > 6:
            correct_option = 1
        df_results.at[index, 'da_cot_ex'] = correct_option


        # print(f"\n\n\n\n---------\n\nQuestion: {question}\n\n\
        # Reference Answer: {system_answer}\n\n\
        # Reasoning: {gen_reasoning}\n\n\
        # Score: {score}\n\
        # Is answer Correct?: {row['human_eval']}\n\n")

    print("Finished evaluating da_cot_ex!")
    df_results.to_csv(csv_path, index=False)
    print('da_cot: ', df_results['da_cot_ex'].mean())

    del(evaluator_pipeline)
    gc.collect()
    torch.cuda.empty_cache()
        


def calculate_mca_with_direct_assessment(csv_path: str, hf_auth: str):
    print("Calculating MCA with Direct Assessment CoT for ", csv_path)
    df_results = pd.read_csv(csv_path)
    df_results['mca_da_cot_reason'] = ""
    df_results['mca_da_cot_score'] = float('nan')
    df_results['mca_da_cot'] = float('nan')

    print("Initializing evaluator model: ")
    try:
        evaluator_pipeline, _, _ = init_model(evaluator_model_id, hf_auth, verbose=True)
    except Exception as e:
        print("Error initializing evaluator model: ", str(e))
        print("Check GPU memory available and try again")
        return

    # Iterate over the rows in the DataFrame
    for index, row in tqdm(df_results.iterrows(), total=len(df_results)):
        # Get the question and correct answer
        reference_text = row['text']
        question = row['question']
        system_answer = row['system_answer']
        choices = [row['correct_answer'], row['wrong_answer1'], row['wrong_answer2'], row['wrong_answer3']]

        ### Max score 10
        mca_scores = []
        for choice in choices:
            prompt = _prompt_da_cot_reason(choice, question, system_answer, max_score=10)
            gen_reasoning = generate_text(evaluator_pipeline, prompt, max_new_tokens=2048)
            prompt = _prompt_da_cot_choose(question, system_answer, gen_reasoning, max_score=10)
            gen_option = generate_text(evaluator_pipeline, prompt, max_new_tokens=4)
            try:
                score = int(gen_option.strip().upper().replace("'", '').replace('"', '').replace("-", '')[0])
            except Exception as e:
                score = -1
            mca_scores.append(score)
        correct_option_score = mca_scores[0]
        correct = True
        for score in mca_scores[1:]:
            if score >= correct_option_score:
                correct = False
        df_results.at[index, 'mca_da_cot'] = 1 if correct else 0


        # print(f"\n\n\n\n---------\n\nQuestion: {question}\n\n\
        # Reference Answer: {system_answer}\n\n\
        # Reasoning: {gen_reasoning}\n\n\
        # Score: {score}\n\
        # Is answer Correct?: {row['human_eval']}\n\n")

    print("Finished evaluating mca_da_cot!")
    df_results.to_csv(csv_path, index=False)
    print('mca_da_cot: ', df_results['mca_da_cot'].mean())

    del(evaluator_pipeline)
    gc.collect()
    torch.cuda.empty_cache()




def _prompt_da_cot_reason_criteria(reference_text, question, system_answer, criteria_description: str, max_score=5):
    prompt = (
        "You are a college professor.\n"
        "Given a base truth text, a question and a student answer give me "
        f"a score from 1 to {max_score} of the answer based on the determined criteria.\n\n"
        "Base truth:\n"
        f"{reference_text}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Candidate Answer:\n"
        f"{system_answer}\n\n"
        f"Evaluate the candidate answer based on the following criteria: {criteria_description}"
        f"Now consider the Base Truth and evaluate the answer based on the criteria, giving a score from 1 to {max_score}."
        "Think step by step and explain your chain of thought. "
    )
    return prompt

def _prompt_da_cot_choose_reasonings(question, system_answer, reasonings: list, max_score=5):
    reasonings = str('\n\n'.join(reasonings))
    prompt = (
        "Given a question, a candidate answer and a professor's evaluation, tell me "
        f"only the final score from 1 to {max_score}.\n\n" 
        "Question:\n"
        f"{question}\n\n"
        "Candidate Answer:\n"
        f"{system_answer}\n\n"
        f"Now based on the following reasonings and evaluations, answer only with the final score from 1 to {max_score}.\n"
        "Reasonings:\n"
        f"{reasonings}"
        "Answer only with the value of the final score and nothing more. Don't try to explain your answer or add any other text."
    )
    return prompt

def calculate_direct_assessment_steps(csv_path: str, hf_auth: str):
    print("Calculating Direct Assessment Steps CoT for ", csv_path)
    df_results = pd.read_csv(csv_path)
    df_results['da_cot_steps_reason'] = ""
    df_results['da_cot_steps_score'] = float('nan')
    df_results['da_cot_steps'] = float('nan')

    print("Initializing evaluator model: ")
    try:
        evaluator_pipeline, _, _ = init_model(evaluator_model_id, hf_auth, verbose=True)
    except Exception as e:
        print("Error initializing evaluator model: ", str(e))
        print("Check GPU memory available and try again")
        return

    # Iterate over the rows in the DataFrame
    for index, row in tqdm(df_results.iterrows(), total=len(df_results)):
        # Get the question and correct answer
        reference_text = row['text']
        question = row['question']
        system_answer = row['system_answer']

        accordance_base_truth = "Accordance to base truth: Evaluate how much of what the candidate answer say is stated on the base truth."
        prompt = _prompt_da_cot_reason_criteria(reference_text, question, system_answer, criteria_description=accordance_base_truth, max_score=10)
        accordance_base_truth_reasoning = generate_text(evaluator_pipeline, prompt, max_new_tokens=2048)
        tech_acc = "Technical Accuracy: Based on what you know and on the base truth, how correct is the candidate answer?"
        prompt = _prompt_da_cot_reason_criteria(reference_text, question, system_answer, criteria_description=tech_acc, max_score=10)
        tech_acc_reasoning = generate_text(evaluator_pipeline, prompt, max_new_tokens=2048)
        answer_relevance = "Answer relevance: is the candidate answer coherent with the question? Is the question fully answered by the candidate?"
        prompt = _prompt_da_cot_reason_criteria(reference_text, question, system_answer,criteria_description=answer_relevance, max_score=10)
        answer_relevance_reasoning = generate_text(evaluator_pipeline, prompt, max_new_tokens=2048)
        prompt = _prompt_da_cot_choose_reasonings(question, system_answer, [accordance_base_truth_reasoning, tech_acc_reasoning, answer_relevance_reasoning], max_score=10)
        gen_option = generate_text(evaluator_pipeline, prompt, max_new_tokens=4)
        # print(">> Generated Option: ", gen_option)
        try:
            score = int(gen_option.strip().upper().replace("'", '').replace('"', '').replace("-", '')[0])
        except Exception as e:
            score = -1

        df_results.at[index, 'da_cot_steps_score'] = score
        df_results.at[index, 'da_cot_steps_reason'] = [accordance_base_truth_reasoning, tech_acc_reasoning, answer_relevance_reasoning]
        correct_option = 0
        if score > 6:
            correct_option = 1
        df_results.at[index, 'da_cot_steps'] = correct_option


        # print(f"\n\n\n\n---------\n\nQuestion: {question}\n\n\
        # Reference Answer: {system_answer}\n\n\
        # Reasoning: {gen_reasoning}\n\n\
        # Score: {score}\n\
        # Is answer Correct?: {row['human_eval']}\n\n")

    print("Finished evaluating da_cot!")
    df_results.to_csv(csv_path, index=False)
    print('da_cot: ', df_results['da_cot'].mean())

    del(evaluator_pipeline)
    gc.collect()
    torch.cuda.empty_cache()




def _prompt_separate_statements(question, answer):
    prompt = (
        "Given the following question and answer:\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:\n"
        f"{answer}\n\n"
        "Divide the answer into one or more statements or affirmations made by the answer. "
        "Please answer only in the following JSON Format:"
"""
{
  "sentences": ["sentence1", "sentence2"]
}
"""
    )
    return prompt

def _prompt_evaluate_sentence(context, sentence):
    prompt = (
        "Given the following context, evaluate the affirmation.\n"
        "Context:\n"
        f"{context}\n\n"
        f"Affirmation:\n{sentence}\n\n"
        "Based on the context, is the affirmation correct or incorrect?\n"
        "Please answer only in the following JSON Format:"
"""
{
  "correct": true or false
}
"""
    )
    return prompt


def calculate_direct_assessment_ragas(csv_path: str, hf_auth: str):
    print("Calculating Direct Assessment RAGAS CoT for ", csv_path)
    df_results = pd.read_csv(csv_path)
    df_results['da_cot_ragas_reason'] = ""
    df_results['da_cot_ragas_score'] = float('nan')
    df_results['da_cot_ragas'] = float('nan')

    print("Initializing evaluator model: ")
    try:
        evaluator_pipeline, _, _ = init_model(evaluator_model_id, hf_auth, verbose=True)
    except Exception as e:
        print("Error initializing evaluator model: ", str(e))
        print("Check GPU memory available and try again")
        return

    # Iterate over the rows in the DataFrame
    for index, row in tqdm(df_results.iterrows(), total=len(df_results)):
        # Get the question and correct answer
        reference_text = row['text']
        question = row['question']
        system_answer = row['system_answer']

        prompt = _prompt_separate_statements(question, system_answer)
        sentences = generate_text(evaluator_pipeline, prompt, max_new_tokens=2048)
        try:
            sentences = sentences.replace("```json", "").replace("```", "").strip()
            json_sentences = json.load(sentences)
        except Exception as e:
            df_results.at[index, 'da_cot_ragas'] = 0
            continue
        
        correct_sentences = 0
        for sentence in json_sentences.get("sentences", []):
            prompt = _prompt_evaluate_sentence(reference_text, sentence)
            gen_option = generate_text(evaluator_pipeline, prompt, max_new_tokens=256)
            try:
                gen_option = gen_option.replace("```json", "").replace("```", "").strip()
                json_gen_option = json.load(gen_option)
            except Exception as e:
                df_results.at[index, 'da_cot_ragas'] = 0
                continue
            if json_gen_option.get("correct", False):
                correct_sentences += 1

        num_sentences = json_sentences.get("sentences", [])
        if num_sentences == 0:
            num_sentences = 1
        df_results.at[index, 'da_cot_ragas_score'] = correct_sentences/num_sentences
        correct_option = 0
        if correct_sentences/num_sentences > 0.5:
            correct_option = 1
        df_results.at[index, 'da_cot_ragas'] = correct_option


        # print(f"\n\n\n\n---------\n\nQuestion: {question}\n\n\
        # Reference Answer: {system_answer}\n\n\
        # Reasoning: {gen_reasoning}\n\n\
        # Score: {score}\n\
        # Is answer Correct?: {row['human_eval']}\n\n")

    print("Finished evaluating da_cot_ragas!")
    df_results.to_csv(csv_path, index=False)
    print('da_cot_ragas: ', df_results['da_cot_ragas'].mean())

    del(evaluator_pipeline)
    gc.collect()
    torch.cuda.empty_cache()




if __name__ == "__main__":
    #Run evaluation for every file in results_dir
    results_dir = 'results/cvpr-papers-glm'
    rag = LcadRag(model_id='THUDM/glm-4-9b-chat')
    for filename in os.listdir(results_dir):
        if filename.endswith(".csv") and not filename.startswith("results"):
            try:
                # Load the dataset with questions and answers
                csv_path = os.path.join(results_dir, filename)
                calculate_multiple_choice_accuracy(rag, csv_path)
            except Exception as e:
                traceback.print_exc()
                print("Error: ", str(e))
                print("Error in evaluating multiple choice questions!!")