import pandas as pd
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import torch
import traceback
try:
    import src.gen_model as gen_model
except:
    import gen_model

import os
from dotenv import load_dotenv

load_dotenv()
hf_auth = os.getenv('HUGGINGFACE_AUTH_TOKEN', '')
data_dir = os.getenv('DATA_DIR', 'data/')


def remove_bad_questions(
            df_questions_path_dir,
            df_questions_name,
            model_pipeline,
            verbose=False
            ):
    """Go through the generated questions and delete them if they are not good questions."""
    df_questions = pd.read_csv(df_questions_path_dir+'/'+df_questions_name)

    SYSTEM_PROMPT =  """Please answer whether the following question is a good question or not.
A bad question is a question that cannot be answered even if you know everything about it. For example, \
"What is the name of the soccer team?" is a bad question, because it isn't possible to know about which team it \
is talking about.
A hard question is not a bad question! It will only be bad if it was not well formulated.

Given the question bellow say whether it is a <bad_question> or a <good_question>.
Explain your chain of thought and then say either <bad_question> or <good_question>.

Question: {question}
A) {correct_answer}
B) {answer2}
C) {answer3}
D) {answer4}

Is a bad_question?
"""
    

    #Create dataframe with prompts
    df_prompts = pd.DataFrame(columns=['prompt'])
    for i in range(len(df_questions.index)):
        line = df_questions.iloc[i]
        question = line['question']
        correct_answer = line['correct_answer']
        wrong_answer1 = line['wrong_answer1']
        wrong_answer2 = line['wrong_answer2']
        wrong_answer3 = line['wrong_answer3']
        initial_prompt = SYSTEM_PROMPT.format(question=question, 
                                              correct_answer=correct_answer, 
                                              answer2=wrong_answer1, 
                                              answer3=wrong_answer2, 
                                              answer4=wrong_answer3)
        try:
            messages = [
                {"role": "system", "content": "Please follow the instructions as accurately as possible."},
                {"role": "user", "content": initial_prompt},
            ]
            prompt = model_pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            prompt = initial_prompt
        df_prompts = pd.concat([df_prompts, pd.DataFrame({'prompt': [prompt]})], ignore_index=True)

    
    dataset = Dataset.from_pandas(df_prompts)
    removed_questions = pd.DataFrame(columns=df_questions.columns)
    indexes_bad_questions = []
    count = 0
    if verbose:
        print(f"Removing bad questions from {df_questions_name}...")
    try:
        pbar = tqdm(total=len(dataset))
        for out in model_pipeline(KeyDataset(dataset, 'prompt'), batch_size=1):
            torch.cuda.empty_cache()
            for seq in out:
                line = df_prompts.iloc[count]
                prompt = line['prompt']
                response = seq['generated_text'][len(prompt):].lower()
                if "bad_question" in response:
                    bad_question = df_questions.iloc[[count]]
                    indexes_bad_questions.append(count)
                    removed_questions = pd.concat([removed_questions, bad_question], ignore_index=True)
                    if verbose:
                        pass
                elif "good_question" not in response:
                    if verbose:
                        pass
                        #print("\n\n----=----\nSem definição: ", seq['generated_text'])
                count+=1
                inc = count - pbar.n
                pbar.update(n=inc)
        df_questions = df_questions.drop(indexes_bad_questions)
        if verbose:
            print("Finished removing bad questions!")
        df_questions.to_csv(df_questions_path_dir+'/'+df_questions_name, index=False)
        removed_questions.to_csv(df_questions_path_dir+'/removed_questions_'+df_questions_name, index=False)
        print(f"Removed {len(removed_questions.index)} bad questions.")
        print(f"Remaining {len(df_questions.index)} questions.")
    except Exception as e:
        traceback.print_exc()
        print("Error: ", str(e))
        print("Erro na remoção das questões ruins.")




if __name__ == "__main__":
    model_pipeline, _, _ =gen_model.init_model(
        model_id="meta-llama/Meta-Llama-3-70B-Instruct",
        hf_auth=hf_auth, 
        verbose=True
    )
    remove_bad_questions(
        df_questions_path_dir=data_dir,
        df_questions_name="df_golden_questions_4kids.csv",
        model_pipeline = model_pipeline,
        verbose = True
    )
    remove_bad_questions(
        df_questions_path_dir=data_dir,
        df_questions_name="df_golden_questions_default.csv",
        model_pipeline = model_pipeline,
        verbose = True
    )
    remove_bad_questions(
        df_questions_path_dir=data_dir,
        df_questions_name="df_golden_questions_pirate.csv",
        model_pipeline = model_pipeline,
        verbose = True
    )
    remove_bad_questions(
        df_questions_path_dir=data_dir,
        df_questions_name="df_golden_questions_scientific.csv",
        model_pipeline = model_pipeline,
        verbose = True
    )
    remove_bad_questions(
        df_questions_path_dir=data_dir,
        df_questions_name="df_golden_questions_technical.csv",
        model_pipeline = model_pipeline,
        verbose = True
    )