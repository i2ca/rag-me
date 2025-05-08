"""Calculate RAGAS metrics"""
try:
    import transformers
    import pandas as pd
    import datasets
    from ragas.metrics import answer_relevancy, faithfulness
    from ragas import RunConfig, evaluate
    try:
        from gen_model import init_model
    except Exception:
        from src.gen_model import init_model
except:
    print("Ragas not activated. Please run `pip install ragas` if you want to calculate RAGAS metrics")


def calculate_ragas_metrics(model_pipeline: transformers.pipeline, result_path: str):
    dataset = generate_ragas_dataset(result_path)
    print(dataset)

    score = evaluate(dataset, 
                     metrics=[faithfulness, answer_relevancy], 
                     run_config=RunConfig(max_workers=5, max_retries=10))
    print("\n------\n>Score:\n", score)
    output_path = result_path.replace(".csv", "_ragas.csv")
    score.to_pandas().to_csv(output_path, index=False)



def generate_ragas_dataset(csv_path: str):
    result_df = pd.read_csv(csv_path)
    # Get only question, system_answer and rag_context
    result_df = result_df[['question', 'system_answer', 'rag_context']]
    # Rename columns to question, answer and contexts
    result_df.columns = ['question', 'answer', 'contexts']
    # Transform contexts column from a string column to a list[str]
    result_df['contexts'] = result_df['contexts'].apply(lambda x: [x])

    # Create dataset
    dataset = datasets.Dataset.from_pandas(result_df)
    return dataset


if __name__ == "__main__":
    import dotenv
    import os

    dotenv.load_dotenv()
    result_path = "results/cvpr-papers-mistral-gte-large/outputs_rag_validation.csv"
    model_pipeline, _, _ = init_model("mistralai/Mistral-7B-Instruct-v0.3", hf_auth=os.environ["HUGGINGFACE_AUTH_TOKEN"])
    calculate_ragas_metrics(model_pipeline, result_path)