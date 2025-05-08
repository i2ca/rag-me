import time
import torch
import transformers
import os
from dotenv import load_dotenv
from transformers.pipelines.pt_utils import KeyDataset
import litellm
import google.generativeai as genai

load_dotenv()

gpu_cuda_device = int(os.getenv('CUDA_DEVICE', 0))
model_revision = os.getenv('GEN_MODEL_REVISION', 'main')

class GeminiModelPipeline():
    call_count = 0
    def __init__(self, model_id) -> None:
        GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=GOOGLE_API_KEY)

        model = genai.GenerativeModel(model_id)
        self.model = model
    def __call__(
            self, 
            query,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            batch_size=1,
            max_new_tokens=None
    ):
        from google.generativeai import GenerationConfig
        if do_sample:
            temperature = 0.9
        else:
            temperature = 0.0
        config = GenerationConfig(max_output_tokens=max_new_tokens, temperature=temperature)
        #check if query is a string
        if isinstance(query, str):
            generated_text = ''
            while generated_text == '':
                try:
                    generated_text = self.model.generate_content(query, generation_config=config).text
                except Exception:
                    generated_text = ''
                    #sleep one second
                    time.sleep(1)
            return [{"generated_text": query+generated_text}]
        elif isinstance(query, list):
            generated_text = ''
            while generated_text == '':
                try:
                    generated_text = self.model.generate_content(query, generation_config=config).text
                    print("\n\n\n>>Generated text: ", generated_text)
                except Exception as e:
                    if "response.text" in str(e):
                        return ""
                    generated_text = ''
                    print("Exception: ", e)
                    #sleep sixty seconds
                    time.sleep(5*60)
            return generated_text
        else:
            query: KeyDataset = query
            for text in query.dataset[query.key]:
                generated_text = ''
                while generated_text == '':
                    try:
                        generated_text = self.model.generate_content(text, generation_config=config).text
                    except Exception:
                        generated_text = ''
                        #sleep one second
                        time.sleep(1)
                # yield [{"generated_text": text+generated_text}]


class OpenAIModelPipeline():
    call_count = 0
    def __init__(self, model_name):
        self.model_name = model_name
        self.api_key = os.getenv('OPENAI_API_KEY')
        if self.api_key is None:
            raise Exception("OPENAI_API_KEY is not set")

    def __call__(
            self, 
            query,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            batch_size=1,
            max_new_tokens=None
    ):
        messages = [{"role": "user", "content": query}]
        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            api_key=self.api_key,
            num_retries=2,
            temperature=0.0,
        )
        content = response["choices"][0]["message"]["content"]
        return [{"generated_text": str(query+content)}]
        

def init_model(model_id, hf_auth, temperature=0.0, verbose=False) -> None:
    if verbose:
        print("Initializing LLM model...")
        print(model_id)
    if 'gemini' in model_id:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            'mistralai/Mistral-7B-Instruct-v0.2',
            token=hf_auth
        )                
        return GeminiModelPipeline(model_id), None, tokenizer
    if 'gpt' in model_id:
        model = OpenAIModelPipeline(model_id)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            'mistralai/Mistral-7B-Instruct-v0.2',
            token=hf_auth
        ) 
        return model, None, tokenizer

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) 
    """ bnb_config = transformers.BitsAndBytesConfig(
        load_in_8bit=True #Use 15Gb of memory
    ) """
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        token=hf_auth,
        trust_remote_code=True,
        revision=model_revision
    )   
    # initialize the model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map="auto", #gpu_cuda_device,
        token=hf_auth,
        # attn_implementation='eager',
        torch_dtype='auto', #torch.bfloat16,
        revision=model_revision
    )
    model.generation_config.top_p=None
    model.eval()
    #Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        token=hf_auth,
        trust_remote_code=True,
        revision=model_revision
    )
    #Initialize pipeline
    do_sample = True
    top_k = 0.9
    if temperature == 0.0:
        temperature = None
        do_sample = False
        top_k = None
    model_pipeline = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        # we pass model parameters here too
        # eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample = do_sample,
        temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        top_k=top_k,
        max_new_tokens=1024,  # mex number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )
    if 'LLAMA' in model_id.upper():
        terminators = [
            model_pipeline.tokenizer.eos_token_id,
            model_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        # model_pipeline.tokenizer.pad_token_id = model.config.eos_token_id # AQUI
        model_pipeline.eos_token_id = terminators # Comment for non llama3 models
    return model_pipeline, model, tokenizer


def generate_text(
    model_pipeline: transformers.pipeline,
    prompt: str,
    max_new_tokens: int = 1024
):
    """Use the model_pipeline to generate an answer to the prompt."""
    try:
        model_pipeline.call_count = 0
        messages = [
            {"role": "user", "content": prompt},
        ]
        try:
            formatted_prompt = model_pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            formatted_prompt = prompt
        return str(model_pipeline(
            formatted_prompt,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
        )[0]['generated_text'][len(formatted_prompt):])
    except Exception as e:
        print(e)