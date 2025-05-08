import pandas as pd
import random

# Read csv
# path = "data/cvpr-papers-test/df_golden_questions_test.csv"
path = "/home/guilhermegoes/projetos/lcad-llm-rag-chat/results/metric_human_eval/manual_eval.csv"
df_questions = pd.read_csv(path)

#for each row, generate a list of integers from 0 to 3 in a random order

df_questions["choices_order"] = df_questions.apply(lambda row: random.sample(range(0, 4), 4), axis=1) 
print(df_questions["choices_order"])

# Save to csv
df_questions.to_csv(path, index=False)

#Read csv
df_questions = pd.read_csv(path)

# Read choices_order from a row and convert to list
row = df_questions.iloc[0]
print(row["choices_order"])
print(type(row["choices_order"]))

choices_order = eval(row["choices_order"])
print(choices_order)
print(type(choices_order))

