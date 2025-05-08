"""We ask the LLM to rewrite the original text with other words so that we can generate the questions from them.
Measure the embedding distance before and after to see if it is right."""

import json
import pandas as pd
import torch
import transformers
import gc
import traceback
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
from src.gen_model import init_model

STYLES = [
    { 
        "style": "default",
        "example": """No example. Just rewrite with other words."""
    },
    { 
        "style": "scientific",
        "example": """Retrieval Augmented Generation (RAG) systems have seen huge popularity in augmenting 
Large-Language Model (LLM) outputs with domain specific and time sensitive data. 
Very recently a shift is happening from simple RAG setups that query a vector database for additional 
information with every user input to more sophisticated forms of RAG. However, different concrete approaches 
compete on mostly anecdotal evidence at the moment. In this paper we present a rigorous dataset creation and 
evaluation workflow to quantitatively compare different RAG strategies. We use a dataset created this way for 
the development and evaluation of a boolean agent RAG setup: A system in which a LLM can decide whether 
to query a vector database or not, thus saving tokens on questions that can be answered with internal 
knowledge. We publish our code and generated dataset online."""
    },
    { 
        "style": "technical",
        "example": """Technical Report: Project Carryall - Nuclear Excavation for Interstate 40 and Atchison, Topeka and Santa Fe Railway

Date: March 1963

Abstract:

Project Carryall, a collaborative initiative between the United States Atomic Energy Commission (AEC) 
and the Atchison, Topeka and Santa Fe Railway (AT&SF), aimed to utilize nuclear explosives to excavate a 
pathway for Interstate 40 and the railway through the Bristol Mountains of southern California. 
This report summarizes the proposal, schedule, and feasibility study for the project..."""
    },
    { 
        "style": "pirate",
        "example": """Ahoy matey, gather 'round and listen well, for I be tellin' ye the tale of linear regression, arrr!
Imagine ye be sailin' the high seas, searchin' fer hidden treasures. 
Now, every treasure map be like a scatterplot, with X marks the spot! But how do ye find the path that leads to the loot? 
That be where linear regression comes in, me hearties."""
    },
    { 
        "style": "4kids",
        "example": """Alright kiddo, let's talk about something called linear regression. 
Imagine you have a bunch of toys scattered on the floor, and you want to clean up the room. 
Linear regression is like figuring out the best way to tidy up by looking at how the toys are spread out.
You see, in math, we often have things called "variables," which are like the different types of toys 
you have. For example, one variable could be how many toy cars you have, and another variable could be 
how many dolls you have. We also have something called the "outcome" or "dependent variable," which is 
like how clean the room is after you pick up all the toys."""
    }
]

def rewrite_in_style(
        df_corpus_original: pd.DataFrame,
        model_pipeline: transformers.pipeline,
        style_name: str,
        example: str,
        verbose: bool = False):
    """Rewrite the original text in a style and returns the generated dataframe."""
    df_new = df_corpus_original.copy()

    SYSTEM_PROMPT =  """Please rewrite the above text in another words in the style: {style}. \
Please mantain the important information and topics of the original text. \
Here is an example of the style:
{example}

Now please rewrite the original text in the style mentioned:
"""
    

    #Create dataframe with prompts
    df_prompts = pd.DataFrame(columns=['prompt'])
    for i in range(len(df_new.index)):
        line = df_new.iloc[i]
        original_text = line['text']
        initial_prompt = "Original text:\n"+original_text+"\n\n"+SYSTEM_PROMPT.format(style=style_name, example=example)
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
    count = 0
    if verbose:
        print(f"Rewriting in style {style_name}...")
    try:
        pbar = tqdm(total=len(dataset))
        for out in model_pipeline(KeyDataset(dataset, 'prompt'), batch_size=1):
            torch.cuda.empty_cache()
            for seq in out:
                line = df_prompts.iloc[count]
                prompt = line['prompt']
                response = seq['generated_text'][len(prompt):]
                
                #print("\n\n-----\n", seq['generated_text'])
                df_new.at[count, 'text'] = response
                count+=1
                inc = count - pbar.n
                pbar.update(n=inc)
        if verbose:
            print("Finished rewriting text!")
        return df_new
    except Exception as e:
        traceback.print_exc()
        print("Error: ", str(e))
        print("Erro na reescrita dos textos.")




def rewrite_original_text(
        df_corpus_path_dir: str,
        model_pipeline: transformers.pipeline,
        df_corpus_name: str="df_corpus_long.csv",
        hf_auth: str = None,
        styles: list = STYLES,
        verbose: bool = False
    ):
    """
    Rewrite the original text with different style and 
    return a list with the filenames of the generated dataframes.
    """

    if verbose:
        print("Rewriting original text from ", df_corpus_path_dir+'/'+df_corpus_name)
    df_corpus_original = pd.read_csv(df_corpus_path_dir+'/'+df_corpus_name)

    my_styles = [style for style in STYLES if style["style"] in styles]

    filenames = []
    for style in my_styles:
        style_name = style["style"]
        example = style["example"]
        new_df = rewrite_in_style(df_corpus_original, model_pipeline, style_name, example, verbose)
        #Save the dataframe
        filename = 'df_corpus_long_'+style_name+'.csv'
        filenames.append(filename)
        new_df.to_csv(df_corpus_path_dir+'/'+filename)
    return filenames

