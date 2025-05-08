import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df_no_rag = pd.read_csv('evaluation/df_rag_False_use_text_False_use_questions_False_use_summary_False.csv')
df_all_rag = pd.read_csv('evaluation/df_rag_True_use_text_True_use_questions_True_use_summary_True.csv')
df_use_text = pd.read_csv('evaluation/df_rag_True_use_text_True_use_questions_False_use_summary_False.csv')
df_use_questions = pd.read_csv('evaluation/df_rag_True_use_text_False_use_questions_True_use_summary_False.csv')
df_use_summary = pd.read_csv('evaluation/df_rag_True_use_text_False_use_questions_False_use_summary_True.csv')

#Load model
embedding_model = SentenceTransformer('thenlper/gte-large')

def evaluate_bert_score(row):
        #Calculate embeddings for all answers
        answers = row[['correct_answer', 'wrong_answer1', 'wrong_answer2', 'wrong_answer3', 'sistem_answer']]
        answers = answers.map(lambda x: embedding_model.encode([x]))
        answers = np.array([x.tolist() for x in answers.to_numpy()]).reshape(5, -1)
        #Calculate similarity between system_answer and all answers
        similarity = cosine_similarity([answers[4]], answers).reshape(-1)
        # Get index of greater similarity
        print(similarity[:-1])
i = 0
while True:
    row = df_all_rag.iloc[i]
    print('\n------------\n\n')
    print(f'Pergunta {i}: ', row['question'])
    print('\nResposta Correta: ', row['correct_answer'])
    print('\nResposta Sistema: ', row['sistem_answer'])
    print('\nBert Score: ', row['bert_score'])
    evaluate_bert_score(row)
    i += 1

    input()
