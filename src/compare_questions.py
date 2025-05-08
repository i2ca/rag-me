"""Compare questions and the original chunks with bert."""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

load_dotenv()
data_dir = os.getenv('DATA_DIR', 'data/default')
embedding_model_name = os.getenv("EMBEDDING_MODEL", "thenlper/gte-large")

embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)

def compare_questions(original_corpus, questions_style, rewritten_corpus, style):
    df_questions = pd.read_csv(questions_style)
    df_original_corpus = pd.read_csv(original_corpus)
    df_rewritten_corpus = pd.read_csv(rewritten_corpus)

    #Join df_questions with df_corpus on text
    df_rewritten_corpus = df_rewritten_corpus.merge(df_questions, on='text', how='left')

    #join both dfs on init_line and source
    df_corpus = df_original_corpus.merge(df_rewritten_corpus, on=['init_line', 'source'], how='left')
    df_corpus = df_corpus.dropna()

    # Get embeddings of question_x and question_y
    question_x_embeddings = embedding_model.encode(list(df_corpus['question']))
    original_text_embeddings = embedding_model.encode(list(df_corpus['text_x']))

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(question_x_embeddings, original_text_embeddings)
    # Get diagonal values of similarity matrix
    similarity = similarity_matrix.diagonal()

    print(f"{style} - Mean Cosine Similarity of questions: ", np.mean(similarity))

if __name__ == "__main__":
    for filename in os.listdir(data_dir):
        if filename.startswith("df_golden_questions"):
            style = filename.split('df_golden_questions_')[-1].split('.')[0]
            original_corpus = f'{data_dir}/df_corpus_long_test.csv'
            rewritten_questions = f'{data_dir}/df_golden_questions_{style}.csv'
            rewritten_corpus = f'{data_dir}/df_corpus_long_{style}.csv'
            compare_questions(original_corpus, rewritten_questions, rewritten_corpus, style)

    
