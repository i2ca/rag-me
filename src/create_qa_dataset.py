"""This script creates a dataset with questions and answers about the text provided in df_corpus_long.csv.
Now, using this chunks, prompts are created and the model is used to generate the answer.
The dataset is saved in a csv file df_qa.csv
"""

import json
import pandas as pd
import torch
import transformers
import gc
import traceback
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
try:
    from src.gen_model import init_model
    from src.create_summary_index import create_summaries
    from src.evaluate_generated_questions import remove_bad_questions
except:
    from gen_model import init_model
    from create_summary_index import create_summaries
    from evaluate_generated_questions import remove_bad_questions
import os
from dotenv import load_dotenv

load_dotenv()
hf_auth = os.getenv('HUGGINGFACE_AUTH_TOKEN', None)
data_dir = os.getenv("DATA_DIR", "")

df_questions_path_dir = 'data'
df_questions_name = 'df_golden_questions.csv'
default_model_id = 'mistralai/Mistral-7B-Instruct-v0.1'

def prompt_llm(
        prompt: str, 
        model_pipeline: transformers.pipeline):
    """Use the model_pipeline to generate an answer to the prompt."""
    try:
        model_pipeline.call_count = 0
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = model_pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )
        return str(model_pipeline(
            prompt,
            num_return_sequences=1,
        )[0]['generated_text'][len(prompt):])
    except Exception as e:
        print(e)

def parse_question_answer(response: str):
        # Format {"question": question, "choices": choices, "gold":index of the correct answer}
        # Return question, correct answer, wrong answer 1, wrong answer 2, wrong answer 3
        try:
            # Localizando a primeira chave '{' e a última chave '}'
            start_index = response.find('{')
            end_index = response.rfind('}') + 1
            # Extraindo a substring que contém o JSON válido
            cleaned_json_string = response[start_index:end_index]
            json_response = json.loads(cleaned_json_string)
            question = json_response['question']
            correct_choice = json_response['correct_choice']
            wrong_choices = json_response['wrong_choices']
            answer = correct_choice
            wrong_answer1 = wrong_choices[0]
            wrong_answer2 = wrong_choices[1]
            wrong_answer3 = wrong_choices[2]
            return question, answer, wrong_answer1, wrong_answer2, wrong_answer3
        except Exception:
            return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

def create_json_questions_answers(
        df_corpus_path_dir: str = df_questions_path_dir,
        df_corpus_name: str="df_corpus_long_test.csv",
        df_questions_path_dir: str = df_questions_path_dir,
        df_questions_name: str = df_questions_name,
        model_pipeline: transformers.pipeline = None,
        hf_auth: str = None,
        verbose: bool = False
    ):
    """Create a dataset with questions and answers about the text provided in df_corpus_long.csv."""
    print("Creating questions and answers...")
    df_corpus = pd.read_csv(df_corpus_path_dir+'/'+df_corpus_name)
    df_questions = pd.DataFrame(columns=['question', 'correct_answer', 'wrong_answer1', 'wrong_answer2', 'wrong_answer3', 'text', 'source'])
    # For each chunk in df_corpus
    for i in tqdm(range(len(df_corpus.index))):
        if verbose:
            print(f'\n----------\n>>>>Processing chunk {i}')
        line = df_corpus.iloc[i]
        text = line['text']
        source = line['source']
        if model_pipeline is None:
            raise ValueError("model_pipeline cannot be None")
        # Create chain of thought explanation
        prompt_cot = ("Considering the following text:\n"+
                      str(text)+"\n\n"+
                      "List the main topics of the above text, explaining each of them.")
        cot_explanation = prompt_llm(prompt_cot, model_pipeline)
        if verbose:
            print("\n>>COT: ", cot_explanation)

        # Create first version of question
        prompt_question = ("Considering the following text:\n"+
                          str(text)+
                          "\nThe main topics of the text are:\n"+
                          str(cot_explanation)+"\n\n"+
                          'You are a professor that wants to create a multiple-choice exam with no notes. \
Given the text above, create one question that can be used as a multiple-choice question, but also \
could be an open ended question, with four answers in JSON format with {"question": question, \
"correct_choice": correct choice, "wrong_choices": [choice0, choice1, choice2]}. Respond without \
markdown and without line breaks. The students will not have access to notes or to the provided \
text during the exam, so it is necessary that the questions \
should be answerable by someone without access to the exact text above and they should not cite the \
given text in any way, like pages, chapters or images. But the question should still be challenging, \
so do not make them too general. Please create only one question.')
        first_question = prompt_llm(prompt_question, model_pipeline)
        if verbose:
            print("\n>>First Question: ", first_question)

        # Criticize the questions
        prompt_question_critics = ("Considering the question created from the following text:\n"+
                                  str(text)+"\n\n"+
                                  "Question in JSON Format: \n"+
                                  str(first_question)+"\n\n"+
                                  'Are there any errors in this question? If there are, '+
                                  'list them and describe each one individually. You do not need to rewrite the question later.')

        critics = prompt_llm(prompt_question_critics, model_pipeline)
        if verbose:
            print("\n>>Critics: ", critics)
        
        # Rewrite the question based on the critics
        prompt_question_improve = ("Considering the question created from the following text:\n"+
                                  str(text)+"\n\n"+
                                  "Question: \n"+
                                  str(first_question)+"\n\n"+
                                  "Considering this list of criticisms about the question:\n"+
                                  str(critics)+"\n\n"+
                                  "Improve the question based on the criticisms. Mantain the question in the "+
                                  "following JSON format:\n\n"+
                                  '{"question": question, '+
                                  '"correct_choice": correct choice, "wrong_choices": [choice0, choice1, choice2]}')

        improved_question = prompt_llm(prompt_question_improve, model_pipeline)
        if verbose:
            print("\n>>Improved Question: ", improved_question)

        # Create dataframe with questions and answers
        (question, answer, wrong_answer1, wrong_answer2, wrong_answer3) = parse_question_answer(improved_question)
        df_questions = pd.concat([df_questions, pd.DataFrame({'question': [question],
                                                              'correct_answer': [answer],
                                                              'wrong_answer1': [wrong_answer1],
                                                              'wrong_answer2': [wrong_answer2],
                                                              'wrong_answer3': [wrong_answer3],
                                                              'text': [text],
                                                              'source': [source]})], ignore_index=True)

    print("Saving questions and answers to "+df_questions_path_dir+'/'+df_questions_name)
    df_questions.to_csv(df_questions_path_dir+'/'+df_questions_name, index=False)


def create_json_questions_answers_simple(
        df_corpus_path_dir: str,
        df_corpus_name: str="df_corpus_long_test.csv",
        df_questions_path_dir: str = df_questions_path_dir,
        df_questions_name: str = df_questions_name,
        style: str = "default",
        model_pipeline: transformers.pipeline = None,
        hf_auth: str = None,
        verbose: bool = False
    ):

    def parse_question_answer(response: str):
        # Format {"question": question, "choices": choices, "gold":index of the correct answer}
        # Return question, correct answer, wrong answer 1, wrong answer 2, wrong answer 3
        try:
            json_response = json.loads(response)
            question = json_response['question']
            correct_choice = json_response['correct_choice']
            wrong_choices = json_response['wrong_choices']
            #answer = choices[json_response['gold']]
            #wrong_answers = choices[:json_response['gold']]+choices[json_response['gold']+1:]
            answer = correct_choice
            wrong_answer1 = wrong_choices[0]
            wrong_answer2 = wrong_choices[1]
            wrong_answer3 = wrong_choices[2]
            return question, answer, wrong_answer1, wrong_answer2, wrong_answer3
        except Exception:
            return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    if verbose:
        print("\n\n----------\nCreating questions and answers from ", df_corpus_path_dir+'/'+df_corpus_name)
    df_corpus = pd.read_csv(df_corpus_path_dir+'/'+df_corpus_name)

    if model_pipeline is None:
        model_pipeline, _, _ = init_model(model_id=default_model_id, hf_auth=hf_auth, verbose=True)

    SYSTEM_PROMPT =  'You are a professor that wants to create a multiple-choice exam with no notes in the style: '+style+'. \
Given the text above, create one question that can be used as a multiple-choice question, but also \
could be an open ended question, with four answers in JSON format with {"question": question, \
"correct_choice": correct choice, "wrong_choices": [choice0, choice1, choice2]}. Respond without \
markdown and without line breaks. The students will not have access to notes or to the provided \
text during the exam, so it is necessary that the questions \
should be answerable by someone without access to the exact text above and they should not cite the \
given text in any way, like pages, chapters or images. But the question should still be challenging, \
so do not make them too general. Please create only one question.'
    STYLE_PROMPT = ' Write the question and answers in the style of the given text: '+style+'. Be exaggerated in the style.'
    if (style != "original") and (style != "test"):
        SYSTEM_PROMPT += STYLE_PROMPT

    #Create dataframe with prompts
    df_prompts = pd.DataFrame(columns=['prompt', 'text'])
    for i in range(len(df_corpus.index)):
        line = df_corpus.iloc[i]
        context = line['text']
        initial_prompt = "Context:\n"+str(context)+"\n\n"+SYSTEM_PROMPT 
        try:
            messages = [
                {"role": "system", "content": "Please try to provide useful, helpful and actionable answers."},
                {"role": "user", "content": initial_prompt},
            ]
            prompt = model_pipeline.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
            )
        except Exception:
            prompt = initial_prompt
        df_prompts = pd.concat([df_prompts, pd.DataFrame({'prompt': [prompt], 'text': [context]})], ignore_index=True)

    #create empty dataframe with columns question, answer, text, init_line
    df_answers = pd.DataFrame(columns=['question', 'correct_answer', 'wrong_answer1', 'wrong_answer2', 'wrong_answer3', 'text'])
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
                text = line['text']
                response = seq['generated_text'][len(prompt):]

                #print("\n\n-------\nPrompt: ", prompt)
                #print(response)
                (question, answer, wrong_answer1, wrong_answer2, wrong_answer3) = parse_question_answer(response)
                #print(f'\n\nQuestion: {question}\nAnswer: {answer}\nWrong Answer 1: {wrong_answer1}\nWrong Answer 2: {wrong_answer2}\nWrong Answer 3: {wrong_answer3}')

                df_answers = pd.concat([df_answers, pd.DataFrame([[question, answer, wrong_answer1, wrong_answer2, wrong_answer3, text]], columns=df_answers.columns)], ignore_index=True)
                count+=1
                inc = count - pbar.n
                pbar.update(n=inc)
        if verbose:
            print("Finished creating questions and answers!")
            print('Number of NA: ', (df_answers['question'].isna()).sum())
        df_answers = df_answers[(df_answers['wrong_answer1'].isna() == False) & \
                                (df_answers['wrong_answer2'].isna() == False) & \
                                (df_answers['wrong_answer3'].isna() == False)]  # noqa: E712
        #save df_answers to csv
        csv_path = df_questions_path_dir+'/'+df_questions_name
        if verbose:
            print('Saving questions and answers to {}'.format(csv_path))
        df_answers.to_csv(csv_path, index=False)
    except Exception as e:
        traceback.print_exc()
        print("Error: ", str(e))
        print("Erro na criação de perguntas e respostas.")


if __name__ == "__main__":
    model_pipeline, _, _ = init_model(model_id="meta-llama/Meta-Llama-3-70B-Instruct", hf_auth=hf_auth, verbose=True)

    create_json_questions_answers(
        df_corpus_path_dir=data_dir,
        df_corpus_name="df_corpus_long_test.csv",
        df_questions_path_dir=data_dir,
        df_questions_name="df_golden_questions_test.csv",
        model_pipeline=model_pipeline,
        hf_auth=hf_auth,
        verbose=True
    )

    model_pipeline, _, _ = init_model(
        model_id="meta-llama/Meta-Llama-3-70B-Instruct", 
        hf_auth=hf_auth, 
        temperature=0.0,
        verbose=True
    )

    remove_bad_questions(
        df_questions_path_dir=data_dir,
        df_questions_name="df_golden_questions_test.csv",
        model_pipeline=model_pipeline,
        verbose=True
    )