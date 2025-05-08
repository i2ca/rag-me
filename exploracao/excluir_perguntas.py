import pandas as pd
import numpy as np

df_questions = pd.read_csv('qa_dataset/df_question_answers.csv')
i = 0
try:
    while(i<len(df_questions.index)):
        print(f"\n\nQUESTION {i}: ")
        print(df_questions.iloc[i]['question'])

        comando = input("\n\nAperte 'X' para deletar a pergunta atual, 'O' para passar para a próxima pergunta e 'Q' para voltar para a última.\n" )
        if comando == 'x' or comando == 'X':
            df_questions.drop([i], inplace=True)
            df_questions.reset_index(inplace=True, drop=True)
        elif comando == 'o' or comando == 'O':
            i+=1
        elif comando == 'q' or comando == 'Q':
            i-=1

        if i%100 == 0:
            df_questions.to_csv('qa_dataset/df_question_answers.csv', index=False)
except Exception as e:
    print(e)
comando = input('Do you want to save? y/n\n')
if comando == 'y':
    #save to csv
    df_questions.to_csv('qa_dataset/df_question_answers.csv', index=False)