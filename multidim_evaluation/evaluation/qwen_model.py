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

def load_qwen_model(llm_arg: str):
    """
    Dynamically loads the Qwen model (7B or 72B) based on the provided llm_arg.

    Parameters
    ----------
    llm_arg : str
        Possible values:
        - 'Qwen2.5_7b'
        - 'Qwen2.5_72b'
        (or any naming scheme you choose)
    """
    global model
    global tokenizer

    model_name = llm_arg

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    print(f"[qwen_model.py] Model loaded for: {llm_arg} -> {model_name}")


def unload_qwen_model():
    """
    Unloads the Llama model and tokenizer from GPU memory.
    This function clears all model-related variables and forces garbage collection
    to free up GPU memory.
    """
    global model
    global tokenizer
    global terminators
    
    try:
        # Delete model from GPU memory
        if 'model' in globals() and model is not None:
            # Move model to CPU first (optional, but can help with cleanup)
            model.cpu()
            # Delete the model
            del model
            print("[qwen_model.py] Model deleted from memory")
        
        # Delete tokenizer
        if 'tokenizer' in globals() and tokenizer is not None:
            del tokenizer
            print("[qwen_model.py] Tokenizer deleted from memory")
            
        # Delete terminators
        if 'terminators' in globals() and terminators is not None:
            del terminators
            print("[qwen_model.py] Terminators deleted from memory")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("[qwen_model.py] GPU cache cleared")
            
        # Force garbage collection
        import gc
        gc.collect()
        print("[qwen_model.py] Garbage collection completed")
        
        # Reset global variables to None
        model = None
        tokenizer = None
        terminators = None
        
        print("[qwen_model.py] Model successfully unloaded from GPU")
        
    except Exception as e:
        print(f"[qwen_model.py] Error during model unloading: {e}")


def generate_response(messages, max_new_tokens=500, do_sample=True, temperature=1.0, top_p=1.0):
    """
    Generates a response using the given messages as input.

    Parameters:
    - messages (list of dict): List of messages in [{"role": ..., "content": ...}, ...] format.
    - max_new_tokens (int, optional): Maximum number of new tokens to generate. Default is 500.
    - do_sample (bool, optional): Whether to use sampling during generation. Default is True.
    - temperature (float, optional): Controls randomness. Higher values lead to more random responses. Default is 1.0.
    - top_p (float, optional): Controls nucleus sampling. Lower values limit diversity. Default is 1.0.

    Returns:
    - str: The generated response.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p
    )
    prompt_length = model_inputs.input_ids.shape[-1]
    generated_ids = output_ids[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


def generate_answer(formatted_prompt):
    if isinstance(formatted_prompt, list):
        formatted_prompt = formatted_prompt[0]
    
    input_messages = [
        {"role": "system", "content": "You are an helpful assistant."},
        {"role": "user", "content": formatted_prompt}
    ]
    return generate_response(input_messages, do_sample=False, temperature=None, top_p=None)


def batch_completion_local(messages):
    responses = []
    for message in messages:
        response = generate_answer(message)
        responses.append(response)
    return responses