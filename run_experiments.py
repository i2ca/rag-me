import os
import subprocess
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
HUGGINGFACE_AUTH_TOKEN = os.getenv("HUGGINGFACE_AUTH_TOKEN", "")
#"GEN_MODEL_REVISION": "04419001bc63e05e70991ade6da1f91c4aeec278",
CUDA_DEVICE = 0
HF_HOME = "/data/huggingface-models/"
# List of environment variables to be updated
# env_variables = [
#     ### Embedding models
#     {
#         "HF_HOME": HF_HOME,
#         "GOOGLE_API_KEY": GOOGLE_API_KEY,
#         "HUGGINGFACE_AUTH_TOKEN": HUGGINGFACE_AUTH_TOKEN,
#         "DATA_DIR":"data/cvpr-papers-all",
#         "TEXT_DIR":"texts/cvpr-papers-500/test",
#         "RESULTS_DIR":"results/paper/dual-base-emb/abs-rag-validation-multilingual",
#         "CUDA_DEVICE":CUDA_DEVICE,
#         "EMBEDDING_MODEL":"intfloat/multilingual-e5-large-instruct",
#         "EMBEDDING_MODEL_DIMENSIONS":1024,
#         "BERT_SCORE_EMBEDDING_MODEL":"thenlper/gte-large",
#         "GEN_MODEL":"meta-llama/Llama-3.1-70B-Instruct",
#         "GEN_MODEL_REVISION":"main",
#         "CHUNK_SIZE":500,
#         "OVERLAP_CHUNKS":50,
#     },
#     {
#         "HF_HOME": HF_HOME,
#         "GOOGLE_API_KEY": GOOGLE_API_KEY,
#         "HUGGINGFACE_AUTH_TOKEN": HUGGINGFACE_AUTH_TOKEN,
#         "DATA_DIR":"data/cvpr-papers-all",
#         "TEXT_DIR":"texts/cvpr-papers-500",
#         "RESULTS_DIR":"results/paper/dual-base-emb/paper-rag-validation-multilingual",
#         "CUDA_DEVICE":CUDA_DEVICE,
#         "EMBEDDING_MODEL":"intfloat/multilingual-e5-large-instruct",
#         "EMBEDDING_MODEL_DIMENSIONS":1024,
#         "BERT_SCORE_EMBEDDING_MODEL":"thenlper/gte-large",
#         "GEN_MODEL":"meta-llama/Llama-3.1-70B-Instruct",
#         "GEN_MODEL_REVISION":"main",
#         "CHUNK_SIZE":500,
#         "OVERLAP_CHUNKS":50,
#     },

#     ### Generation Models Abstracts
#     {
#         "HF_HOME": HF_HOME,
#         "GOOGLE_API_KEY": GOOGLE_API_KEY,
#         "HUGGINGFACE_AUTH_TOKEN": HUGGINGFACE_AUTH_TOKEN,
#         "DATA_DIR":"data/cvpr-papers-all",
#         "TEXT_DIR":"texts/cvpr-papers-500/test",
#         "RESULTS_DIR":"results/paper/dual-base/abs-rag-validation-gemma1",
#         "CUDA_DEVICE":CUDA_DEVICE,
#         "EMBEDDING_MODEL":"intfloat/multilingual-e5-large-instruct",
#         "EMBEDDING_MODEL_DIMENSIONS":1024,
#         "BERT_SCORE_EMBEDDING_MODEL":"thenlper/gte-large",
#         "GEN_MODEL":"google/gemma-7b-it",
#         "GEN_MODEL_REVISION":"main",
#         "CHUNK_SIZE":500,
#         "OVERLAP_CHUNKS":50,
#     },

#     {
#         "HF_HOME": HF_HOME,
#         "GOOGLE_API_KEY": GOOGLE_API_KEY,
#         "HUGGINGFACE_AUTH_TOKEN": HUGGINGFACE_AUTH_TOKEN,
#         "DATA_DIR":"data/cvpr-papers-all",
#         "TEXT_DIR":"texts/cvpr-papers-500/test",
#         "RESULTS_DIR":"results/paper/dual-base/abs-rag-validation-gemma2",
#         "CUDA_DEVICE":CUDA_DEVICE,
#         "EMBEDDING_MODEL":"intfloat/multilingual-e5-large-instruct",
#         "EMBEDDING_MODEL_DIMENSIONS":1024,
#         "BERT_SCORE_EMBEDDING_MODEL":"thenlper/gte-large",
#         "GEN_MODEL":"google/gemma-2-27b-it",
#         "GEN_MODEL_REVISION":"main",
#         "CHUNK_SIZE":500,
#         "OVERLAP_CHUNKS":50,
#     },

#     {
#         "HF_HOME": HF_HOME,
#         "GOOGLE_API_KEY": GOOGLE_API_KEY,
#         "HUGGINGFACE_AUTH_TOKEN": HUGGINGFACE_AUTH_TOKEN,
#         "DATA_DIR":"data/cvpr-papers-all",
#         "TEXT_DIR":"texts/cvpr-papers-500/test",
#         "RESULTS_DIR":"results/paper/dual-base/abs-rag-validation-llama2",
#         "CUDA_DEVICE":CUDA_DEVICE,
#         "EMBEDDING_MODEL":"intfloat/multilingual-e5-large-instruct",
#         "EMBEDDING_MODEL_DIMENSIONS":1024,
#         "BERT_SCORE_EMBEDDING_MODEL":"thenlper/gte-large",
#         "GEN_MODEL":"meta-llama/Llama-2-7b-chat-hf",
#         "GEN_MODEL_REVISION":"main",
#         "CHUNK_SIZE":500,
#         "OVERLAP_CHUNKS":50,
#     },

#     {
#         "HF_HOME": HF_HOME,
#         "GOOGLE_API_KEY": GOOGLE_API_KEY,
#         "HUGGINGFACE_AUTH_TOKEN": HUGGINGFACE_AUTH_TOKEN,
#         "DATA_DIR":"data/cvpr-papers-all",
#         "TEXT_DIR":"texts/cvpr-papers-500/test",
#         "RESULTS_DIR":"results/paper/dual-base/abs-rag-validation-llama3",
#         "CUDA_DEVICE":CUDA_DEVICE,
#         "EMBEDDING_MODEL":"intfloat/multilingual-e5-large-instruct",
#         "EMBEDDING_MODEL_DIMENSIONS":1024,
#         "BERT_SCORE_EMBEDDING_MODEL":"thenlper/gte-large",
#         "GEN_MODEL":"meta-llama/Llama-3.1-70B-Instruct",
#         "GEN_MODEL_REVISION":"main",
#         "CHUNK_SIZE":500,
#         "OVERLAP_CHUNKS":50,
#     },

#     {
#         "HF_HOME": HF_HOME,
#         "GOOGLE_API_KEY": GOOGLE_API_KEY,
#         "HUGGINGFACE_AUTH_TOKEN": HUGGINGFACE_AUTH_TOKEN,
#         "DATA_DIR":"data/cvpr-papers-all",
#         "TEXT_DIR":"texts/cvpr-papers-500/test",
#         "RESULTS_DIR":"results/paper/dual-base/abs-rag-validation-qwen1.5",
#         "CUDA_DEVICE":CUDA_DEVICE,
#         "EMBEDDING_MODEL":"intfloat/multilingual-e5-large-instruct",
#         "EMBEDDING_MODEL_DIMENSIONS":1024,
#         "BERT_SCORE_EMBEDDING_MODEL":"thenlper/gte-large",
#         "GEN_MODEL":"Qwen/Qwen1.5-7B-Chat",
#         "GEN_MODEL_REVISION":"main",
#         "CHUNK_SIZE":500,
#         "OVERLAP_CHUNKS":50,
#     },

#     {
#         "HF_HOME": HF_HOME,
#         "GOOGLE_API_KEY": GOOGLE_API_KEY,
#         "HUGGINGFACE_AUTH_TOKEN": HUGGINGFACE_AUTH_TOKEN,
#         "DATA_DIR":"data/cvpr-papers-all",
#         "TEXT_DIR":"texts/cvpr-papers-500/test",
#         "RESULTS_DIR":"results/paper/dual-base/abs-rag-validation-qwen2.5",
#         "CUDA_DEVICE":CUDA_DEVICE,
#         "EMBEDDING_MODEL":"intfloat/multilingual-e5-large-instruct",
#         "EMBEDDING_MODEL_DIMENSIONS":1024,
#         "BERT_SCORE_EMBEDDING_MODEL":"thenlper/gte-large",
#         "GEN_MODEL":"Qwen/Qwen2.5-32B-Instruct",
#         "GEN_MODEL_REVISION":"main",
#         "CHUNK_SIZE":500,
#         "OVERLAP_CHUNKS":50,
#     },

#     ### Generation Models Papers
#     {
#         "HF_HOME": HF_HOME,
#         "GOOGLE_API_KEY": GOOGLE_API_KEY,
#         "HUGGINGFACE_AUTH_TOKEN": HUGGINGFACE_AUTH_TOKEN,
#         "DATA_DIR":"data/cvpr-papers-all",
#         "TEXT_DIR":"texts/cvpr-papers-500",
#         "RESULTS_DIR":"results/paper/dual-base/paper-rag-validation-gemma1",
#         "CUDA_DEVICE":CUDA_DEVICE,
#         "EMBEDDING_MODEL":"intfloat/multilingual-e5-large-instruct",
#         "EMBEDDING_MODEL_DIMENSIONS":1024,
#         "BERT_SCORE_EMBEDDING_MODEL":"thenlper/gte-large",
#         "GEN_MODEL":"google/gemma-7b-it",
#         "GEN_MODEL_REVISION":"main",
#         "CHUNK_SIZE":500,
#         "OVERLAP_CHUNKS":50,
#     },

#     {
#         "HF_HOME": HF_HOME,
#         "GOOGLE_API_KEY": GOOGLE_API_KEY,
#         "HUGGINGFACE_AUTH_TOKEN": HUGGINGFACE_AUTH_TOKEN,
#         "DATA_DIR":"data/cvpr-papers-all",
#         "TEXT_DIR":"texts/cvpr-papers-500",
#         "RESULTS_DIR":"results/paper/dual-base/paper-rag-validation-gemma2",
#         "CUDA_DEVICE":CUDA_DEVICE,
#         "EMBEDDING_MODEL":"intfloat/multilingual-e5-large-instruct",
#         "EMBEDDING_MODEL_DIMENSIONS":1024,
#         "BERT_SCORE_EMBEDDING_MODEL":"thenlper/gte-large",
#         "GEN_MODEL":"google/gemma-2-27b-it",
#         "GEN_MODEL_REVISION":"main",
#         "CHUNK_SIZE":500,
#         "OVERLAP_CHUNKS":50,
#     },

#     {
#         "HF_HOME": HF_HOME,
#         "GOOGLE_API_KEY": GOOGLE_API_KEY,
#         "HUGGINGFACE_AUTH_TOKEN": HUGGINGFACE_AUTH_TOKEN,
#         "DATA_DIR":"data/cvpr-papers-all",
#         "TEXT_DIR":"texts/cvpr-papers-500",
#         "RESULTS_DIR":"results/paper/dual-base/paper-rag-validation-llama2",
#         "CUDA_DEVICE":CUDA_DEVICE,
#         "EMBEDDING_MODEL":"intfloat/multilingual-e5-large-instruct",
#         "EMBEDDING_MODEL_DIMENSIONS":1024,
#         "BERT_SCORE_EMBEDDING_MODEL":"thenlper/gte-large",
#         "GEN_MODEL":"meta-llama/Llama-2-7b-chat-hf",
#         "GEN_MODEL_REVISION":"main",
#         "CHUNK_SIZE":500,
#         "OVERLAP_CHUNKS":50,
#     },

#     {
#         "HF_HOME": HF_HOME,
#         "GOOGLE_API_KEY": GOOGLE_API_KEY,
#         "HUGGINGFACE_AUTH_TOKEN": HUGGINGFACE_AUTH_TOKEN,
#         "DATA_DIR":"data/cvpr-papers-all",
#         "TEXT_DIR":"texts/cvpr-papers-500",
#         "RESULTS_DIR":"results/paper/dual-base/paper-rag-validation-llama3",
#         "CUDA_DEVICE":CUDA_DEVICE,
#         "EMBEDDING_MODEL":"intfloat/multilingual-e5-large-instruct",
#         "EMBEDDING_MODEL_DIMENSIONS":1024,
#         "BERT_SCORE_EMBEDDING_MODEL":"thenlper/gte-large",
#         "GEN_MODEL":"meta-llama/Llama-3.1-70B-Instruct",
#         "GEN_MODEL_REVISION":"main",
#         "CHUNK_SIZE":500,
#         "OVERLAP_CHUNKS":50,
#     },

#     {
#         "HF_HOME": HF_HOME,
#         "GOOGLE_API_KEY": GOOGLE_API_KEY,
#         "HUGGINGFACE_AUTH_TOKEN": HUGGINGFACE_AUTH_TOKEN,
#         "DATA_DIR":"data/cvpr-papers-all",
#         "TEXT_DIR":"texts/cvpr-papers-500",
#         "RESULTS_DIR":"results/paper/dual-base/paper-rag-validation-qwen1.5",
#         "CUDA_DEVICE":CUDA_DEVICE,
#         "EMBEDDING_MODEL":"intfloat/multilingual-e5-large-instruct",
#         "EMBEDDING_MODEL_DIMENSIONS":1024,
#         "BERT_SCORE_EMBEDDING_MODEL":"thenlper/gte-large",
#         "GEN_MODEL":"Qwen/Qwen1.5-7B-Chat",
#         "GEN_MODEL_REVISION":"main",
#         "CHUNK_SIZE":500,
#         "OVERLAP_CHUNKS":50,
#     },

#     {
#         "HF_HOME": HF_HOME,
#         "GOOGLE_API_KEY": GOOGLE_API_KEY,
#         "HUGGINGFACE_AUTH_TOKEN": HUGGINGFACE_AUTH_TOKEN,
#         "DATA_DIR":"data/cvpr-papers-all",
#         "TEXT_DIR":"texts/cvpr-papers-500",
#         "RESULTS_DIR":"results/paper/dual-base/paper-rag-validation-qwen2.5",
#         "CUDA_DEVICE":CUDA_DEVICE,
#         "EMBEDDING_MODEL":"intfloat/multilingual-e5-large-instruct",
#         "EMBEDDING_MODEL_DIMENSIONS":1024,
#         "BERT_SCORE_EMBEDDING_MODEL":"thenlper/gte-large",
#         "GEN_MODEL":"Qwen/Qwen2.5-32B-Instruct",
#         "GEN_MODEL_REVISION":"main",
#         "CHUNK_SIZE":500,
#         "OVERLAP_CHUNKS":50,
#     },

# ]

env_variables = []
chunk_sizes = [100, 500, 1000, 5000]
chunk_overlaps = [0, 25, 50, 75]
for chunk_size in chunk_sizes:
    for chunk_overlap in chunk_overlaps:
        env_variables.append(
            {
                "HF_HOME": "/data/huggingface-models/",
                "GOOGLE_API_KEY": GOOGLE_API_KEY,
                "HUGGINGFACE_AUTH_TOKEN": HUGGINGFACE_AUTH_TOKEN,
                "DATA_DIR":'data/cvpr-papers-all',
                "TEXT_DIR":'texts/cvpr-papers-500',
                "RESULTS_DIR":f'results/paper/parameters-llama-multilingual/cvpr-papers-{chunk_size}-{chunk_overlap}',
                "CUDA_DEVICE":CUDA_DEVICE,
                "EMBEDDING_MODEL":'intfloat/multilingual-e5-large-instruct',
                "EMBEDDING_MODEL_DIMENSIONS":1024,
                "GEN_MODEL":"meta-llama/Meta-Llama-3-8B-Instruct",
                "GEN_MODEL_REVISION": "main",
                "BERT_SCORE_EMBEDDING_MODEL": "thenlper/gte-large",
                "CHUNK_SIZE":chunk_size,
                "OVERLAP_CHUNKS": chunk_overlap
            }
        )

# Path to the .env file
env_file_path = ".env"

# Function to update the .env file
def update_env_file(env_vars):
    with open(env_file_path, 'w') as file:
        for key, value in env_vars.items():
            file.write(f"{key}={value}\n")
            # Set env var
            os.environ[key] = str(value)

# Function to run evaluate.py script
def run_evaluate_script(run_all: bool = False, run_chat: bool = False):
    if run_all:
        subprocess.run(['python', 'init_rag.py'], stdout=None, stderr=None)
        subprocess.run(['python', 'chat_main.py'], stdout=None, stderr=None)
    elif run_chat:
        subprocess.run(['python', 'chat_main.py'], stdout=None, stderr=None)
    subprocess.run(['python', 'evaluate_chatbot_answers.py'], stdout=None, stderr=None)


# Main logic
last_emb_model = ""
last_text_dir = ""
last_chunk_size = 0
last_chunk_overlap = -1
for env_vars in tqdm(env_variables, desc="Total Progress Experiments"):
    print("\n\n>>>>>>>>Running Evaluation with: ", str(env_vars), "\n\n")
    update_env_file(env_vars)
    load_dotenv()
    # Create results directory if it doesn't exist
    if not os.path.exists(env_vars["RESULTS_DIR"]):
        os.makedirs(env_vars["RESULTS_DIR"])
    # Save env configs in results dir
    with open(f'{env_vars["RESULTS_DIR"]}/configurations.txt', 'w') as f:
        f.write(f'"DATA_DIR":"{env_vars["DATA_DIR"]}",\n'),
        f.write(f'"TEXT_DIR":"{env_vars["TEXT_DIR"]}",\n'),
        f.write(f'"RESULTS_DIR":"{env_vars["RESULTS_DIR"]}",\n'),
        f.write(f'"CUDA_DEVICE":"{env_vars["CUDA_DEVICE"]}",\n'),
        f.write(f'"EMBEDDING_MODEL":"{env_vars["EMBEDDING_MODEL"]}",\n'),
        f.write(f'"EMBEDDING_MODEL_DIMENSIONS":{env_vars["EMBEDDING_MODEL_DIMENSIONS"]},\n'),
        f.write(f'"BERT_SCORE_EMBEDDING_MODEL":"{env_vars["BERT_SCORE_EMBEDDING_MODEL"]}",\n'),
        f.write(f'"GEN_MODEL":"{env_vars["GEN_MODEL"]}",\n'),
        f.write(f'"GEN_MODEL_REVISION":"{env_vars["GEN_MODEL_REVISION"]}",\n'),
        f.write(f'"CHUNK_SIZE":{env_vars["CHUNK_SIZE"]},\n'),
        f.write(f'"OVERLAP_CHUNKS":{env_vars["OVERLAP_CHUNKS"]},\n')
    
    
    # Run evaluate.py
    if env_vars["EMBEDDING_MODEL"] != last_emb_model or env_vars["TEXT_DIR"] != last_text_dir or env_vars["CHUNK_SIZE"] != last_chunk_size or env_vars["OVERLAP_CHUNKS"] != last_chunk_overlap:
        run_evaluate_script(run_all=True)
        last_emb_model = env_vars["EMBEDDING_MODEL"]
        last_text_dir = env_vars["TEXT_DIR"]
        last_chunk_size = env_vars["CHUNK_SIZE"]
        last_chunk_overlap = env_vars["OVERLAP_CHUNKS"]
    else:
        run_evaluate_script(run_chat=True)
