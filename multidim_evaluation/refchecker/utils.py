import os
import warnings

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore', message='.*generation flags.*')


import json
import spacy

# from .llm_module.llama_model import *
# load_llama_model('meta-llama/Llama-3.3-70B-Instruct')
# load_llama_model('meta-llama/Llama-3.1-8B-Instruct')

from .llm_module.qwen_model import *
# load_qwen_model('Qwen/Qwen3-Next-80B-A3B-Instruct')
load_qwen_model('Qwen/Qwen3-30B-A3B-Instruct-2507')

# Setup spaCy NLP
nlp = None

# Setup OpenAI API
openai_client = None

# Setup Claude 2 API
bedrock = None
anthropic_client = None

 
def sentencize(text):
    """Split text into sentences"""
    global nlp
    if not nlp:
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent for sent in doc.sents]


def split_text(text, segment_len=200):
    """Split text into segments according to sentence boundaries."""
    segments, seg = [], []
    sents = [[token.text for token in sent] for sent in sentencize(text)]
    for sent in sents:
        if len(seg) + len(sent) > segment_len:
            segments.append(" ".join(seg))
            seg = sent
            # single sentence longer than segment_len
            if len(seg) > segment_len:
                # split into chunks of segment_len
                seg = [
                    " ".join(seg[i:i+segment_len])
                    for i in range(0, len(seg), segment_len)
                ]
                segments.extend(seg)
                seg = []
        else:
            seg.extend(sent)
    if seg:
        segments.append(" ".join(seg))
    return segments


def get_model_batch_response(
        prompts,
        model,
        temperature=0,
        n_choices=1,
        max_new_tokens=500,
        api_base=None,
        sagemaker_client=None,
        sagemaker_params=None,
        sagemaker_get_response_func=None,
        custom_llm_api_func=None,
        **kwargs
):
    """
    Get batch generation results with given prompts.

    Parameters
    ----------
    prompts : List[str]
        List of prompts for generation.
    temperature : float, optional
        The generation temperature, use greedy decoding when setting
        temperature=0, defaults to 0.
    model : str, optional
        The model for generation, defaults to 'bedrock/anthropic.claude-3-sonnet-20240229-v1:0'.
    n_choices : int, optional
        How many samples to return for each prompt input, defaults to 1.
    max_new_tokens : int, optional
        Maximum number of newly generated tokens, defaults to 500.

    Returns
    -------
    response_list : List[str]
        List of generated text.
    """
    if not prompts or len(prompts) == 0:
        raise ValueError("Invalid input.")
    
    if sagemaker_client is not None:
        parameters = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature
        }
        if sagemaker_params is not None:
            for k, v in sagemaker_params.items():
                if k in parameters:
                    parameters[k] = v
        response_list = []
        for prompt in prompts:
            r = sagemaker_client.invoke_endpoint(
                EndpointName=model,
                Body=json.dumps(
                    {
                        "inputs": prompt,
                        "parameters": parameters,
                    }
                ),
                ContentType="application/json",
            )
            if sagemaker_get_response_func is not None:
                response = sagemaker_get_response_func(r)
            else:
                r = json.loads(r['Body'].read().decode('utf8'))
                response = r['outputs'][0]
            response_list.append(response)
        return response_list
    
    elif custom_llm_api_func is not None:
        return custom_llm_api_func(prompts)
    else:
        message_list = []
        for prompt in prompts:
            if len(prompt) == 0:
                raise ValueError("Invalid prompt.")
            if isinstance(prompt, str):
                messages = [prompt]
            elif isinstance(prompt, list):
                messages = prompt
            else:
                raise ValueError("Invalid prompt type.")
            message_list.append(messages)

        while True:
            responses = batch_completion_local(
                messages=message_list,
            )
            response_list = responses
            return response_list
