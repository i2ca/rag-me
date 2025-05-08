import src.create_summary_index as create_summary_index
import src.create_question_index as create_question_index
import src.create_faiss_index as create_faiss_index
import src.create_qa_dataset as create_qa_dataset
from src.rewrite_original_text import rewrite_original_text
from src.evaluate_generated_questions import remove_bad_questions
from dotenv import load_dotenv
import src.gen_model as gen_model
import os

load_dotenv()

#Flags
json_questions = True
rewrite = False #####

#Create golden standard question-answers
data_dir = os.getenv('DATA_DIR', 'data/default')
text_dir = os.getenv('TEXT_DIR', 'texts/default')
hf_auth = os.getenv('HUGGINGFACE_AUTH_TOKEN')

test_text_dir = text_dir+'/test'

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

#Initialize llm model
model_pipeline, _, _ = gen_model.init_model(
    model_id="meta-llama/Meta-Llama-3-70B-Instruct",
    hf_auth=hf_auth, 
    verbose=True
)

# create_faiss_index.pdf_to_text(texts_path_dir=test_text_dir, verbose=True)
print()
create_faiss_index.create_df_corpus(
    texts_path_dir = text_dir, 
    df_corpus_path_dir = data_dir, 
    df_corpus_name="df_corpus_long_original.csv",
    paragraph_size = 4000, 
    paragraph_overlap = 40, 
    #sample_size = 50,
    verbose = True
)
print()
create_faiss_index.create_df_corpus(
    texts_path_dir = test_text_dir, 
    df_corpus_path_dir = data_dir, 
    df_corpus_name="df_corpus_long_test.csv",
    paragraph_size = 4000, 
    paragraph_overlap = 40, 
    #sample_size = 50,
    verbose = True
)
print()

if rewrite:
    rewrite_original_text(
        df_corpus_path_dir=data_dir,
        model_pipeline = model_pipeline,
        df_corpus_name = "df_corpus_long_original.csv",
        hf_auth = hf_auth,
        styles = ["default", "pirate", "scientific", "4kids", "technical"],
        verbose = True
    )

texts_list = os.listdir(data_dir)
texts_list = [filename for filename in texts_list if filename.startswith('df_corpus_long_')]

# Go through texts and create questions
for filename in texts_list:
    results_filename = f"df_golden_questions_{filename.split('df_corpus_long_')[1]}"
    if not json_questions:
        create_question_index.create_questions(
            df_corpus_path_dir=data_dir,
            df_corpus_name=filename,
            df_questions_path_dir=data_dir,
            df_questions_name=results_filename,
            model_pipeline=model_pipeline,
            verbose = True
        )
        print()
        create_qa_dataset.create_wrong_answers(
            df_questions_path_dir = data_dir,
            df_questions_name = results_filename,
            model_pipeline = model_pipeline,
            verbose = True
        )
    else: #Create json questions
        create_qa_dataset.create_json_questions_answers(
            df_corpus_path_dir=data_dir,
            df_corpus_name=filename,
            df_questions_path_dir=data_dir,
            df_questions_name=results_filename,
            #style=filename.split('df_corpus_long_')[1].split('.')[0],
            model_pipeline=model_pipeline,
            hf_auth=hf_auth,
            verbose = True
        )
        remove_bad_questions(
            df_questions_path_dir=data_dir,
            df_questions_name=results_filename,
            model_pipeline=model_pipeline,
            verbose=True
        )
