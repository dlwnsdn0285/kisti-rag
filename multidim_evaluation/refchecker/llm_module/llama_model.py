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



def unload_llama_model():
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
            print("[llama_model.py] Model deleted from memory")
        
        # Delete tokenizer
        if 'tokenizer' in globals() and tokenizer is not None:
            del tokenizer
            print("[llama_model.py] Tokenizer deleted from memory")
            
        # Delete terminators
        if 'terminators' in globals() and terminators is not None:
            del terminators
            print("[llama_model.py] Terminators deleted from memory")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("[llama_model.py] GPU cache cleared")
            
        # Force garbage collection
        import gc
        gc.collect()
        print("[llama_model.py] Garbage collection completed")
        
        # Reset global variables to None
        model = None
        tokenizer = None
        terminators = None
        
        print("[llama_model.py] Model successfully unloaded from GPU")
        
    except Exception as e:
        print(f"[llama_model.py] Error during model unloading: {e}")
        

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

def batch_completion_local(messages):
    responses = []
    for message in messages:
        response = generate_answer(message)
        responses.append(response)
    return responses