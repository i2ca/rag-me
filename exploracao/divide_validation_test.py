import pandas as pd


df = pd.read_csv('data/cvpr-papers-500-sergio/df_golden_questions_test_valid.csv')

# Create 40% validation 60% test
df_test = df.sample(frac=0.6, random_state=42)
df_valid = df.drop(df_test.index)

df_test.to_csv('data/cvpr-papers-test/df_golden_questions_test.csv', index=False)
df_valid.to_csv('data/cvpr-papers-validation/df_golden_questions_validation.csv', index=False)