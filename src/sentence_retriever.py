#Semantic search using sbert and FAISS

from pypdf import PdfReader
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from faiss import write_index, read_index
import time
from tqdm import tqdm
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from dotenv import load_dotenv

load_dotenv()

embedding_model_name = os.getenv("EMBEDDING_MODEL", "thenlper/gte-large")
hf_token = os.getenv("HUGGINGFACE_AUTH_TOKEN", "")

chunks_index_name = 'chunks.index'
questions_index_name = 'questions.index'
summary_index_name = 'summary.index'
chunks_df_name = 'df_corpus.csv'
questions_df_name = 'df_questions.csv'
summary_df_name = 'df_summary.csv'
dir = 'data'


def unique(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]

class SentenceRetriever:
    def __init__(self, encoder_model_id: str = embedding_model_name, data_dir: str = dir):
        self.model = ''
        self.df_corpus = ''
        self.paragraphs_per_search = 3
        self.model = SentenceTransformer(encoder_model_id, trust_remote_code=True, token=hf_token)

        try:
            self.index_chunks = read_index(data_dir+'/'+chunks_index_name)
            self.df_corpus = pd.read_csv(data_dir+'/'+chunks_df_name)
        except Exception:
            print("Could not load chunks index. Please run init_rag.py first.")
        try:
            self.df_questions = pd.read_csv(data_dir+'/'+questions_df_name)
            self.index_questions = read_index(data_dir+'/'+questions_index_name)
        except Exception:
            self.index_questions = None
            self.df_questions = None
        try:
            self.df_summary = pd.read_csv(data_dir+'/'+summary_df_name)
            self.index_summary = read_index(data_dir+'/'+summary_index_name)
        except Exception:
            self.index_summary = None
            self.df_summary = None


    def fetch_text(self, dataframe_idx):
        def merge_strings(str1, str2):
            str1 = str(str1)
            str2 = str(str2)
            # Find the common overlap
            overlap = find_overlap(str1, str2)
            # Concatenate the strings, excluding the common overlap
            return str1 + str2[len(overlap):]
        def find_overlap(str1, str2):
            max_overlap = min(len(str1), len(str2))
            overlap = ""
            # Iterate through possible overlaps
            for i in range(1, max_overlap + 1):
                if str1.endswith(str2[:i]):
                    overlap = str2[:i]
            return overlap
        if(self.paragraphs_per_search == 1):
            info = self.df_corpus.iloc[dataframe_idx]
            return info.text
        else:
            info = self.df_corpus.iloc[dataframe_idx-(int(self.paragraphs_per_search/2)):dataframe_idx+int(self.paragraphs_per_search/2)+1]
        text = ""
        for info1 in info.text:
            text = merge_strings(text, info1)
        return text
    
    def fetch_qa(self, dataframe_idx):
        row = self.df_questions.iloc[dataframe_idx]
        #Create a string with the question and the answer
        qa = "Questionary extracted from the books:\n{} {}".format(row.question, row.answer)

        return qa

    def fetch_summary(self, dataframe_idx):
        row = self.df_summary.iloc[dataframe_idx]
        return row.summary
        
    def search_query(self, query, top_k, previous_context = [], rag_use_text = True, rag_use_questions = True, rag_use_summary = True, verbose = False):
        t=time.time()
        results = []
        query_vector = self.model.encode([query])

        if rag_use_text:
            top_k_results_text = self.index_chunks.search(query_vector, 3*top_k)     
            #Process results from raw text
            top_k_ids_text = unique(top_k_results_text[1].tolist()[0])
            #Verify if top_k_ids_text have indexes close to each other
            for i in range(len(top_k_ids_text)):
                for j in range(0, i):
                    if abs(top_k_ids_text[i] - top_k_ids_text[j]) <= self.paragraphs_per_search:
                        top_k_ids_text[i] = top_k_ids_text[j]
            #Verify if the context is in previous_context
            i = 0
            while i < len(top_k_ids_text):
                text = self.fetch_text(top_k_ids_text[i])
                if text in previous_context:
                    top_k_ids_text = np.delete(top_k_ids_text, i)
                else:
                    i += 1
            #clean top_k_ids_text 
            top_k_ids_text = unique(top_k_ids_text)[:top_k]
            results_text = [self.fetch_text(idx) for idx in top_k_ids_text]

            results += results_text

        if rag_use_questions:
            top_k_results_questions = self.index_questions.search(query_vector, 3*top_k)
            #Process results from questions
            top_k_ids_questions = unique(top_k_results_questions[1].tolist()[0])
            #Verify if top_k_ids_questions have repeated questions
            for i in range(len(top_k_ids_questions)):
                for j in range(0, i):
                    if self.df_questions.iloc[i].question == self.df_questions.iloc[j].question:
                        top_k_ids_questions[i] = top_k_ids_questions[j]
            #Verify if the context is in previous_context
            i = 0
            while i < len(top_k_ids_questions):
                text = self.fetch_qa(top_k_ids_questions[i])
                if text in previous_context:
                    top_k_ids_questions = np.delete(top_k_ids_questions, i)
                else:
                    i += 1
            top_k_ids_questions = unique(top_k_ids_questions)[:top_k]
            results_questions = [self.fetch_qa(idx) for idx in top_k_ids_questions]
            results += results_questions

        if rag_use_summary:
            top_k_results_summary = self.index_summary.search(query_vector, 3*top_k)
            #Process results from summary
            top_k_ids_summary = unique(top_k_results_summary[1].tolist()[0])
            #Verify if the context is in previous_context
            i = 0
            while i < len(top_k_ids_summary):
                text = self.fetch_summary(top_k_ids_summary[i])
                if text in previous_context:
                    top_k_ids_summary = np.delete(top_k_ids_summary, i)
                else:
                    i += 1
            top_k_ids_summary = unique(top_k_ids_summary)[:top_k]
            results_summaries = [self.fetch_summary(idx) for idx in top_k_ids_summary]
            results += results_summaries

        #LOGS
        #print cossine distance for each top_k_ids_text and query
        #for i in range(len(top_k_ids_text)):
            #get vector top_k_ids_text[i] in the index
            #vec = [self.index_chunks.index.reconstruct(int(top_k_ids_text[i]))]
            #cos_similarity = cosine_similarity(query_vector, vec)
            #print(f'{i} - Cosine Similarity: {top_k_ids_text[i]} - {cos_similarity[0][0]}')
            #print(f'{top_k_ids_text[i]} - Source: {self.df_corpus.iloc[top_k_ids_text[i]].source}')

        if verbose:
            print('>>>> Results in Total Time: {}'.format(time.time()-t))

        return results

    def get_context_subquestions(self, sub_questions, query, model_pipeline, rag_use_text = True, \
                              rag_use_questions = True, rag_use_summary = True, verbose = False):
        """Get the context for each subquery."""
        prompt = """<s>[INST] Could you produce a brief summary of the provided text while ensuring it retains the necessary information to address the given question? This summary should encapsulate the essential details relevant to the question at hand, facilitating a comprehensive response.
Question: {sub_question}
Text: {context}
[INST]
"""
        raw_context = []
        previous_raw_context = []
        context = []
        df_prompts = pd.DataFrame(columns=['prompt'])
        for sub_question in sub_questions:
            raw_context = self.search_query(sub_question, 1, previous_context = previous_raw_context, rag_use_text = rag_use_text, \
                              rag_use_questions = rag_use_questions, rag_use_summary = rag_use_summary, verbose = verbose)
            df_prompts = pd.concat([df_prompts, pd.DataFrame({'prompt': [prompt.format(context='\n'.join(raw_context), 
                                                                                       sub_question=sub_question)]})], ignore_index=True)
            previous_raw_context += raw_context
        dataset_prompts = Dataset.from_pandas(df_prompts)
        try:
            #pbar = tqdm(total=len(dataset_prompts))
            model_pipeline.call_count = 0
            count=0
            for out in model_pipeline(KeyDataset(dataset_prompts, 'prompt'), batch_size=1):
                torch.cuda.empty_cache()
                for seq in out:
                    prompt = df_prompts.iloc[count]['prompt']
                    response = seq['generated_text'][len(prompt):]
                    sub_question = sub_questions[count]
                    if verbose:
                        print("Sub Question: ", sub_question)
                        print("Summary: ", response)
                    context += [sub_question + '\n' + response]
                    count+=1
                    #inc = count - pbar.n
                    #pbar.update(n=inc)
        except Exception as e:
            print("Error: ", str(e))
            print("Error in creating context summaries!!")
        return context

    def get_context(
            self, 
            query, 
            model_pipeline, 
            rag_use_text = True, 
            rag_use_questions = True, 
            rag_use_summary = True,
            rag_query_expansion = True, 
            verbose = False):
        """Expands the original query and return relevant context."""
        def parse_sub_questions(response):
            return [question.strip() for question in response.split('\n')]
        if rag_query_expansion:
            # First prompt the model to expand the original query
            prompt = f"""<s>[INST] Given the following user question, give me sub questions that would give me the information needed to answer the main question.
    You don't need to add the information itself, only the sub questions that are needed to answer the user question.
    All the questions must be independent of each other, so they should be understood by themselves.
    User question: {query}
    [INST]
    """
            model_pipeline.call_count = 0
            sequences = model_pipeline(
                prompt,
                num_return_sequences=1,
            )
            torch.cuda.empty_cache()
            generated_text = sequences[0]['generated_text']
            response = generated_text[len(prompt):]  # Remove the prompt from the output
            sub_questions = parse_sub_questions(response)
            if verbose:
                print("\n\nQuery expansion:\n\n", sub_questions)
            #For each sub_question, search 1 chunk, question, summary
            context = self.get_context_subquestions(sub_questions, query, model_pipeline, rag_use_text = rag_use_text, \
                                rag_use_questions = rag_use_questions, rag_use_summary = rag_use_summary, verbose = verbose)
            #Add rag for the original query
            context += self.search_query(query, 1, previous_context = context, rag_use_text = rag_use_text, \
                        rag_use_questions = rag_use_questions, rag_use_summary = rag_use_summary, verbose = verbose)
            return context
        if not rag_query_expansion:
            return self.search_query(query, 3, previous_context = [], rag_use_text = rag_use_text, \
                        rag_use_questions = rag_use_questions, rag_use_summary = rag_use_summary, verbose = verbose)



if __name__ == "__main__":
    retriever = SentenceRetriever()
    while(1):
        query = str(input("Query: "))
        #What are the main ingredients for a blast furnace?
        results = retriever.search_query(query, 2)
        for result in results:
            print("\n\n---------\n\n")
            print(result) 





