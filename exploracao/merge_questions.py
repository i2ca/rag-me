import pandas as pd
import numpy as np

"""
df_questions = pd.read_csv('qa_dataset/df_question_answers.csv')

df_no_rag = 'evaluation/df_rag_False_use_text_False_use_questions_False_use_summary_False.csv'
df_all_rag = 'evaluation/df_rag_True_use_text_True_use_questions_True_use_summary_True.csv'
df_use_text = 'evaluation/df_rag_True_use_text_True_use_questions_False_use_summary_False.csv'
df_use_questions = 'evaluation/df_rag_True_use_text_False_use_questions_True_use_summary_False.csv'
df_use_summary = 'evaluation/df_rag_True_use_text_False_use_questions_False_use_summary_True.csv'

path = df_no_rag

df =  pd.read_csv(path)
df = df[df['init_line'].isin(df_questions['init_line'])]
df.reset_index(inplace=True, drop=True)
df['text'] = df_questions['text']
print(df.info())
df.to_csv(path)

path = df_all_rag

df =  pd.read_csv(path)
df = df[df['init_line'].isin(df_questions['init_line'])]
df.reset_index(inplace=True, drop=True)
df['text'] = df_questions['text']
print(df.info())
df.to_csv(path)

path = df_use_text

df =  pd.read_csv(path)
df = df[df['init_line'].isin(df_questions['init_line'])]
df.reset_index(inplace=True, drop=True)
df['text'] = df_questions['text']
print(df.info())
df.to_csv(path)

path = df_use_questions

df =  pd.read_csv(path)
df = df[df['init_line'].isin(df_questions['init_line'])]
df.reset_index(inplace=True, drop=True)
df['text'] = df_questions['text']
print(df.info())
df.to_csv(path)

path = df_use_summary

df =  pd.read_csv(path)
df = df[df['init_line'].isin(df_questions['init_line'])]
df.reset_index(inplace=True, drop=True)
df['text'] = df_questions['text']
print(df.info())
df.to_csv(path)
"""


df_questions =  pd.read_csv('qa_dataset/antigo/df_question_answers2024111.csv')

df = pd.read_csv('evaluation/df_rag_True_use_text_False_use_questions_False_use_summary_True.csv')
df_questions = df_questions[df_questions['init_line'].isin(df['init_line'])]
df_questions.reset_index(inplace=True, drop=True)
df_questions['text'] = df['text']
print(df_questions.info())
df_questions.to_csv('qa_dataset/df_question_answers.csv', index=False)