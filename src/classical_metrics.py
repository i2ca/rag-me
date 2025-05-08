import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge 

def calculate_bert_score(embedding_model: SentenceTransformer, result_path):
    print("Calculating BERT scores...")
    ## Comparação com BERTlike
    #Primeira estratégia para avaliação das respostas. Iremos avaliar a similaridade do 
    # embedding da resposta dada pelo sistema com 4 respostas possíveis. 
    # Uma dessas respostas está correta e as outras 3 estão erradas. 
    # Iremos dizer que o modelo acertou se o embedding mais próximo for da resposta correta, e errou se não for.

    def evaluate_bert_score(row):
        #Calculate embeddings for all answers
        answers = row[['correct_answer', 'wrong_answer1', 'wrong_answer2', 'wrong_answer3', 'system_answer']]
        answers = answers.map(lambda x: embedding_model.encode([str(x)]))
        answers = np.array([x.tolist() for x in answers.to_numpy()]).reshape(5, -1)
        #Calculate similarity between system_answer and all answers
        similarity = cosine_similarity([answers[4]], answers).reshape(-1)
        # Get index of greater similarity
        greater_similarity_index = np.argmax(similarity[:-1])
        # See if it is the correct answer
        if greater_similarity_index == 0:
            return 1
        else:
            return 0
    dataset = pd.read_csv(result_path)
    dataset['bert_score'] = dataset.apply(evaluate_bert_score, axis=1)
    dataset.to_csv(result_path, index=False)
    print(f"Saved BERT scores to {result_path}")	
    print('BERT Score: ', dataset['bert_score'].mean())

def calculate_rouge_score(result_path):
    print("Calculating Rouge-L Recall Scores...")
    def evaluate_rouge_score(row):
        #Calculate rouge-L between correct_answer and system_answer
        answers = row[['correct_answer', 'system_answer']].to_list()
        rouge = Rouge()
        score = rouge.get_scores(str(answers[1]), str(answers[0]))[0]['rouge-l']['r']
        #print(score)
        return score
    dataset = pd.read_csv(result_path)
    dataset['rouge_score'] = dataset.apply(evaluate_rouge_score, axis=1)
    dataset.to_csv(result_path, index=False)
    print(f"Saved Rouge L scores to {result_path}")	
    print('Recall Rouge-L Score: ', dataset['rouge_score'].mean())