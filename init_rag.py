import src.create_summary_index as create_summary_index
import src.create_question_index as create_question_index
import src.create_faiss_index as create_faiss_index
import src.create_qa_dataset as create_qa_dataset
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import src.gen_model as gen_model
import os

load_dotenv()

#Flag
init_complete_rag = False # Set to true to create questions an summaries

data_dir = os.getenv('DATA_DIR', 'data/default')
text_dir = os.getenv('TEXT_DIR', 'texts/default')

hf_auth = os.getenv('HUGGINGFACE_AUTH_TOKEN')

embedding_model_name = os.getenv("EMBEDDING_MODEL", "thenlper/gte-large")
embedding_model_dimensions = int(os.getenv("EMBEDDING_MODEL_DIMENSIONS", 1024))
model_id = os.getenv("GEN_MODEL", 'mistralai/Mistral-7B-Instruct-v0.2')

chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
overlap_chunks = int(os.getenv("OVERLAP_CHUNKS", 80))

if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)

print("Initializing RAG indexes with embedding model", embedding_model_name)

#Initialize encoder model
print("Initing SentenceTransformers")
encoder_model = SentenceTransformer(embedding_model_name, trust_remote_code=True, token=hf_auth, device="cuda")
encoder_model_dimensions = embedding_model_dimensions #Update when changing the model

#Reading pdfs in text_dir and creating df_corpus
create_faiss_index.pdf_to_text(texts_path_dir=text_dir, verbose=True)
print("Creating df_corpus")
print()
create_faiss_index.create_df_corpus(
    texts_path_dir = text_dir, 
    df_corpus_path_dir = data_dir, 
    paragraph_size = chunk_size, 
    paragraph_overlap = overlap_chunks, 
    verbose = True
)
print()

print("Creating Index")
#Creating chunks index
create_faiss_index.create_index(
    df_corpus_path_dir=data_dir,
    index_dir = data_dir,
    encoder_model=encoder_model,
    encoder_model_dimensions=encoder_model_dimensions,
    verbose = True
)
print()

if init_complete_rag:
    #Initialize llm model
    model_pipeline, _, _ = gen_model.init_model(
        model_id=model_id, 
        hf_auth=hf_auth, 
        verbose=True
    )
    create_faiss_index.create_df_corpus(
        texts_path_dir = text_dir, 
        df_corpus_path_dir = data_dir, 
        df_corpus_name="df_corpus_long.csv", #This one will be used to create the questions and summaries for the RAG
        paragraph_size = 4000, 
        paragraph_overlap = 80, 
        verbose = True
    )
    print()

    #Creating questions index
    create_question_index.create_questions(
        df_corpus_path_dir=data_dir,
        df_questions_path_dir=data_dir,
        model_pipeline=model_pipeline,
        verbose = True
    )
    print()
    create_question_index.create_index(
        encoder_model=encoder_model,
        encoder_model_dimensions=encoder_model_dimensions,
        df_questions_path_dir=data_dir,
        index_dir = data_dir,
        verbose = True
    )
    print()
    #Creating summaries index
    create_summary_index.create_summaries(
        df_corpus_path_dir=data_dir,
        df_summary_path_dir=data_dir,
        model_pipeline=model_pipeline,
        verbose = True
    )
    print()
    create_summary_index.create_index(
        encoder_model=encoder_model,
        encoder_model_dimensions=encoder_model_dimensions,
        df_summary_path_dir=data_dir,
        index_dir = data_dir,
        verbose = True
    )
    print()