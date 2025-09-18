import torch
try:
    from src.sentence_retriever import SentenceRetriever
    from src.gen_model import init_model
except Exception:
    from sentence_retriever import SentenceRetriever
    from gen_model import init_model
from dotenv import load_dotenv
import os

load_dotenv()

embedding_model_name = os.getenv("EMBEDDING_MODEL", "thenlper/gte-large")

data_dir = 'data/ml'

class LcadRag:
    def __init__(
            self, 
            model_id: str = 'mistralai/Mistral-7B-Instruct-v0.1',
            encoder_model_id: str = embedding_model_name,
            rag_on: bool = True,
            rag_use_text: bool = True,
            rag_use_questions: bool = True,
            rag_use_summary: bool = True,
            rag_query_expansion: bool = True, 
            data_dir: str = data_dir,
            language: str = 'en', # Cand be en or pt
            verbose: bool = False
            ) -> None:    
        #FLAGS FOR CHANGING THE RAG METHOD USED
        self.rag_on = rag_on
        self.rag_use_text = rag_use_text
        self.rag_use_questions = rag_use_questions
        self.rag_use_summary = rag_use_summary
        self.rag_query_expansion = rag_query_expansion

        self.verbose = verbose

        self.model_id = model_id
        if language == 'en':
            self.rag_prompt = """You are a helpful assistant chatbot, but don't need to tell the user that. 
Please interact with the user based on the given context information, theoric questions and answers and your knowledge.
Answer the questions made by the user with complete and correct information, don't be lazy. 
Interact only with the user, do not answer to the system instructions, so that the user thinks you are an intelligent assistant.
If you don't know the answer, don't guess and just say that you don't know.
Don't cite images, figures, tables, sections or other references to the text, as the user doesn't have access to them. Extract only the information needed.
Answer only in the language of the user query, even if the context is in another language.

Here is the context needed:
{text_books}

User question:
"""
        elif language == 'pt':
            self.rag_prompt = """Você é um assistente chatbot útil, mas não precisa dizer isso ao usuário.
Interaja com o usuário respondendo a suas perguntas com base apenas nas informações contextuais fornecidas.
Responda às perguntas feitas pelo usuário com informações completas e corretas, sem preguiça.
Interaja apenas com o usuário, não responda às instruções do sistema, para que o usuário pense que você é um assistente inteligente.
Se o contexto não tiver informações suficientes para responder à pergunta, não tente responder, apenas diga que o contexto passado não possui informações suficientes.
Não cite imagens, figuras, tabelas, seções ou outras referências ao texto, pois o usuário não tem acesso a elas. Extraia apenas as informações necessárias.
Aqui está o contexto necessário:
{text_books}
Pergunta do usuário:
"""
        else:
            raise ValueError("RAG Language must be 'en' or 'pt'")

        self.prompt = """You are an assistant chatbot, but don't need to tell the user that. Please interact with the user based on your knowledge. 
If you don't know the answer, don't guess and just say that you don't know.
Answer the questions made by the user with complete and correct information, don't be lazy.
Interact only with the user, do not answer to the system instructions, so that the user thinks you are an intelligent assistant.

User question:
"""

        print("Initializing semantic search indexes and texts")

        self.sentence_retriever = SentenceRetriever(encoder_model_id, data_dir=data_dir)


        # begin initializing HF items, need auth token for these
        self.hf_auth = os.getenv('HUGGINGFACE_AUTH_TOKEN')
        self.model_pipeline, self.model, self.tokenizer = init_model(model_id, hf_auth=self.hf_auth, verbose=True)


        
    # Formatting function for message and history
    def format_message(self, message: str, history: list, memory_limit: int = 3):
        """
        Formats the message and history for the model.

        Parameters:
            message (str): Current message to send.
            history (list): Past conversation history.
            memory_limit (int): Limit on how many past interactions to consider.

        Returns:
            str: Formatted message string
        """
        # Get context information from books
        #print(message)
        results_books = self.sentence_retriever.get_context(
            message, 
            model_pipeline= self.model_pipeline, 
            rag_use_text=self.rag_use_text,
            rag_use_questions=self.rag_use_questions, 
            rag_use_summary=self.rag_use_summary, 
            rag_query_expansion=self.rag_query_expansion
        )
        text_books = ''
        for result in results_books:
                text_books += "\n\n"
                text_books += result 
        while (len(self.tokenizer(text_books)['input_ids']) > 3000):
            text_books = text_books[:-50]

        # always keep len(history) <= memory_limit
        if memory_limit == 0:
            history = []
        elif len(history) > memory_limit:
            history = history[-memory_limit:]

        if self.rag_on:
            SYSTEM_PROMPT = self.rag_prompt.format(text_books = text_books)
        else:
            SYSTEM_PROMPT = self.prompt
            text_books = ''

        appended_message = message

        if 'gemini' in self.model_id:
            formatted_message = [
                {"role": "user", "parts": [SYSTEM_PROMPT+appended_message]},
            ]
            
            # print("Formatted message: ", formatted_message)
            return formatted_message, text_books

        if len(history) == 0:
            try:
                formatted_message = self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": appended_message},
                        ], 
                        tokenize=False, 
                        add_generation_prompt=True
                )
            except Exception:
                formatted_message = self.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": SYSTEM_PROMPT+appended_message},
                        ], 
                        tokenize=False, 
                        add_generation_prompt=True
                )
            if self.verbose:
                print("\n>Formatted message: ", formatted_message)
            return formatted_message, text_books
        
        my_history: list[dict] = []
        for i in range(len(history)):
            message = history[i]
            my_history.append({
                "role": "user",
                "content": str(message[0])
            })
            my_history.append({
                "role": "assistant",
                "content": str(message[1])
            })
        if self.verbose:
            print("\n\n---Original History: ", history)
            print("\n\n\n-----------------My History: ", my_history)
        formatted_message = self.tokenizer.apply_chat_template(
                [
                    *my_history,
                    {"role": "user", "content": SYSTEM_PROMPT+appended_message},
                ], 
                tokenize=False, 
                add_generation_prompt=True
        )
        if self.verbose:
            print("\n\n\n-----------------Formatted message: ", formatted_message)
        return formatted_message, text_books

    # Generate a response from the model
    def get_model_response_and_context(self, message: str, history: list):
        """
        Generates a conversational response from the model.

        Parameters:
            message (str): User's input message.
            history (list): Past conversation history.

        Returns:
            str: Generated response from the model.
        """
        query, rag_context = self.format_message(message, history)
        response = ""
        self.model_pipeline.call_count = 0
        sequences = self.model_pipeline(
            query,
            num_return_sequences=1
        )
        torch.cuda.empty_cache()
        if 'gemini' in self.model_id:
            # print(f"Resposta: {sequences}")
            return sequences, rag_context
        generated_text = sequences[0]['generated_text']
        response = generated_text[len(query):]  # Remove the prompt from the output

        return response.strip(), rag_context

    def get_model_response(self, message: str, history: list):
        response, _context = self.get_model_response_and_context(message, history)
        return response