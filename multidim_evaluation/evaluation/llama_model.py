import os
import warnings

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore', message='.*generation flags.*')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import random 

# Set random seeds for reproducibility
def set_seed():
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

set_seed()

def load_llama_model(llm_arg: str):
    """
    Loads the Llama model (8B or 70B) based on the provided llm_arg string.
    
    Parameters
    ----------
    llm_arg : str
        Possible values:
        - 'Llama3.1_8b'
        - 'Llama3.1_70b'
        (or any other naming scheme you desire)
    """
    global model
    global tokenizer
    global terminators

    # 1) Determine model_name based on llm_arg
    model_name = llm_arg

    # 2) Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 3) Handle terminators and special tokens
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    special_tokens_dict = {"pad_token": "<pad>", "eos_token": "</s>"}
    tokenizer.add_special_tokens(special_tokens_dict)

    print(f"[llama_model.py] Model loaded for: {llm_arg} -> {model_name}")




def generate_response(messages, max_tokens=512, do_sample=True, temperature=0.6, top_p=0.9):
    """
    Generates a response using the given messages as input.

    Parameters:
    - messages (list of dict): List of messages in [{"role": ..., "content": ...}, ...] format.
    - max_new_tokens (int, optional): Maximum number of new tokens to generate. Default is 512.
    - do_sample (bool, optional): Whether to use sampling during generation. Default is True.
    - temperature (float, optional): Controls randomness. Higher values lead to more random responses. Default is 0.6.
    - top_p (float, optional): Controls nucleus sampling. Lower values limit diversity. Default is 0.9.

    Returns:
    - str: The generated response.
    """
    input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        eos_token_id = terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
        temperature=temperature,
        top_p = top_p
    )    
    
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def generate_answer(formatted_prompt):
    input_messages = [
        {"role": "system", "content": "You are an helpful assistant."},
        {"role": "user", "content": formatted_prompt}
    ]
    return generate_response(input_messages, do_sample=False, temperature=None, top_p=None)

def batch_completion_local(model,messages,temperature):
    responses = []
    for message in messages:
        response = generate_answer(message)
        responses.append(response)
    return responses