import torch
import transformers
#import gradio as gr
from dotenv import load_dotenv
import os

load_dotenv()

print("Initializing model llama2 70b")
model_id = 'meta-llama/Llama-2-70b-chat-hf'
# begin initializing HF items, need auth token for these
hf_auth = os.getenv('HUGGINGFACE_AUTH_TOKEN')

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
) 

""" bnb_config = transformers.BitsAndBytesConfig(
    load_in_8bit=True #Use 15Gb of memory
) """

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    token=hf_auth
)   

# initialize the model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    token=hf_auth
)
model.eval()

#Initialize tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    token=hf_auth
)

#Initialize pipeline
llama_pipeline = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    # we pass model parameters here too
    temperature=0.3,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=1024,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

 
# Mais simples
while(0):
    user_prompt = str(input("Send a message: "))
    prompt = user_prompt
    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=50,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    ) 
    for seq in sequences:
        print(f"Response: {seq['generated_text'][len(prompt):]}")

    print("\n\n") 


#Utilizando GRADIO

SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You are a helpful AI assistant.
<</SYS>>

"""


# Formatting function for message and history
def format_message(message: str, history: list, memory_limit: int = 2):
    """
    Formats the message and history for the Llama model.

    Parameters:
        message (str): Current message to send.
        history (list): Past conversation history.
        memory_limit (int): Limit on how many past interactions to consider.

    Returns:
        str: Formatted message string
    """
    # Get context information from books

    # always keep len(history) <= memory_limit
    if len(history) > memory_limit:
        history = history[-memory_limit:]

    appended_message = message

    if len(history) == 0:
        formatted_message = SYSTEM_PROMPT + f"{appended_message} [/INST]"
        print(f"Formated message:\n{formatted_message}")
        return formatted_message

    formatted_message = SYSTEM_PROMPT + f"{history[0][0]} [/INST] {history[0][1]} </s>"

    # Handle conversation history
    for user_msg, model_answer in history[1:]:
        formatted_message += f"<s>[INST] {user_msg} [/INST] {model_answer} </s>"

    # Handle the current message
    formatted_message += f"<s>[INST] {appended_message} [/INST]"

    #print(f"Formated message:\n{formatted_message}")

    return formatted_message

# Generate a response from the Llama model
def get_llama_response(message: str, history: list) -> str:
    """
    Generates a conversational response from the Llama model.

    Parameters:
        message (str): User's input message.
        history (list): Past conversation history.

    Returns:
        str: Generated response from the Llama model.
    """
    query = format_message(message, history)
    response = ""

    sequences = llama_pipeline(
        query,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    torch.cuda.empty_cache()

    generated_text = sequences[0]['generated_text']
    response = generated_text[len(query):]  # Remove the prompt from the output

    final_response += "\n\n\nResponse by the bot:\n\n"+response.strip()
    return response.strip()

#gr.ChatInterface(get_llama_response).launch(share=True)
