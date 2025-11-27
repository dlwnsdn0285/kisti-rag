# rewrite_chatgpt_query_expansion_llama3.py
import json
import os
os.environ["TRANSFORMERS_CACHE"] = "/data/../llm_cache"
# import openai
import jsonlines
from tqdm import tqdm
import time
import argparse
import io
import json

# longlora: inference-qlora.py
import sys
# Add the directory containing the target file to sys.path
sys.path.append('/home/../Rag/LongLoRA/')
import math
import torch
import textwrap
import transformers
from peft import PeftModel
from transformers import GenerationConfig, TextStreamer, BitsAndBytesConfig
from llama_attn_replace import replace_llama_attn

# model_name = "gpt-3.5-turbo-0125"
# openai.api_key = ''

PROMPT_DICT = {
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_input_llama2": (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
    "prompt_llama2": "[INST]{instruction}[/INST]"
}

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def set_prompt(line, args, n_recent=3, tokenizer=None):
    
    Instruction = "Given a question, please answer the question in a sentence. The answer should be as informative as possible."

    if args.prompt_type == "icl":
        e1 = """Question: and by whom was the game the last of us established?\nAnswer: Andy Gavin and Jason Rubin. Naughty Dog, LLC (formerly JAM Software, Inc.) is an American first−party video game developer based in Santa Monica, California. Founded by Andy Gavin and Jason Rubin in 1984 as an independent developer."""
        e2 = """Question: is chelsea a club?\nAnswer: Yes, chelsea is an English professional football club."""
        e3 = """Question: is call me by your name a movie?\nAnswer: Yes, based on a book of the same name. Call Me by Your Name is a 2017 coming−of− age romantic drama film directed by Luca Guadagnino. Its screenplay, by James Ivory, who also co−produced, is based on the 2007 novel of the same name by Andr Aciman."""
        e4 = """Question: where was ulysses s. grant from?\nAnswer: Hiram Ulysses Grant was born in Point Pleasant, Ohio, on April 27, 1822, to Jesse Root Grant, a tanner and merchant, and Hannah Simpson Grant."""
    
        prompt = f"{Instruction}\n\n{e1}\n\n{e2}\n\n{e3}\n\n{e4}\n\nQuestion: {line['output']}\nAnswer: "
        
    # return prompt

    if args.use_llama3_1:
        prompt = [{'content':prompt, 'role':'user'}]
        prompt = apply_chat_template_content(prompt, tokenizer)
    else: # use llama2+longlora
        prompt_no_input = PROMPT_DICT["prompt_llama2"]
        prompt = prompt_no_input.format_map({"instruction": prompt})
    # print("prompt: ", prompt)
    return prompt


def apply_chat_template_content(prompt, tokenizer):
    messages = prompt
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    # add = "<|start_header_id|>assistant<|end_header_id|>"
    return tokenizer.apply_chat_template(messages, tokenize=False) # + add


def generate_rewrite(line, respond, args, tokenizer=None):
    prompt = set_prompt(line, args, tokenizer=tokenizer)

    output = respond(prompt=prompt)
    return output

def build_lamma(args):
    if args.use_llama3_1:
        # if args.sft_model: 
        #     model_id = args.sft_model
        # else: 
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        response_template = "<|start_header_id|>assistant<|end_header_id|>"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
)
        
        # Load the base model
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id,
                                                    torch_dtype= "auto",
                                                    device_map="auto", 
                                                    quantization_config=quantization_config,
                                                    cache_dir=args.cache_dir,
                                                    )
        # # Load the adapter model
        # model = PeftModel.from_pretrained(model, adapter_model_name)


        # Load the tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id,
                                                               cache_dir=args.cache_dir,)

        # Define special tokens
        special_tokens_dict = {
            "additional_special_tokens": [response_template],
        }

        # Add the special tokens to the tokenizer
        tokenizer.add_special_tokens(special_tokens_dict)

        # Resize the model embeddings to accommodate the new special tokens
        model.resize_token_embeddings(len(tokenizer))

        model.eval()
        
        respond = build_generator_llama3_1(model, tokenizer, temperature=args.temperature,
                            top_p=args.top_p, max_gen_len=args.max_gen_len, use_cache=True)
        return respond, tokenizer
    
    else: # ! llama2+longlora 
        if args.flash_attn:
            replace_llama_attn(inference=True)

        if args.sft_model:  
            model_id = args.sft_model
            
            # Set RoPE scaling factor
            config = transformers.AutoConfig.from_pretrained(
                model_id,
                cache_dir=args.cache_dir,
            )

            orig_ctx_len = getattr(config, "max_position_embeddings", None)
            if orig_ctx_len and args.context_size > orig_ctx_len:
                scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
                config.rope_scaling = {"type": "linear", "factor": scaling_factor}

            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                cache_dir=args.cache_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            )
            adapter_model_name = args.base_model
            model = PeftModel.from_pretrained(model, adapter_model_name)

        else:
                
            # Set RoPE scaling factor
            config = transformers.AutoConfig.from_pretrained(
                args.base_model,
                cache_dir=args.cache_dir,
            )

            orig_ctx_len = getattr(config, "max_position_embeddings", None)
            if orig_ctx_len and args.context_size > orig_ctx_len:
                scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
                config.rope_scaling = {"type": "linear", "factor": scaling_factor}

            # Load model and tokenizer
            model = transformers.AutoModelForCausalLM.from_pretrained(
                args.base_model,
                config=config,
                cache_dir=args.cache_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            )


        model.resize_token_embeddings(32001)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
            padding_side="right",
            use_fast=False,
        )

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        respond = build_generator(model, tokenizer, temperature=args.temperature, top_p=args.top_p,
                                max_gen_len=args.max_gen_len, use_cache=True)
        return respond

def build_generator(
    model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True
):
    def response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        streamer = TextStreamer(tokenizer)
        
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            streamer=streamer,
        )
        
        out = tokenizer.decode(output[0], skip_special_tokens=True)

        out = out.split(prompt.lstrip("<s>"))[1].strip()
        return out

    return response

def build_generator_llama3_1(
    model, tokenizer,  temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True
):
    def response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        streamer = TextStreamer(tokenizer)
        
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            streamer=streamer,
            eos_token_id=terminators,
        )
        
        out = tokenizer.decode(output[0],)
        response_template = tokenizer.special_tokens_map['additional_special_tokens'][0]
        out = remove_special_tokens(out.split(response_template)[1].strip(), tokenizer)
        # out = out.split(prompt.lstrip("<s>"))[1].strip()

        return out

    return response

def remove_special_tokens(text, tokenizer):
    for token in tokenizer.special_tokens_map.values():
        if isinstance(token, list):
            token = token[0]
        text = text.replace(token, "")
    # for token in tokenizer.additional_special_tokens:
    #     text = text.replace(token, "")
    return text.strip()

if __name__ == "__main__":
    # args setup #######
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, default=None) # e.g., 'test-modified-sampled.json'
    parser.add_argument('--output_root', type=str, default=None) # e.g., 
    parser.add_argument('--output_path', type=str, default=None) # e.g., 
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--prompt_type', type=str, default='zsl') # e.g., 'icl', 'zsl', etc.
    parser.add_argument("--use_pssg", action="store_true", help="use pssg information")
    parser.add_argument("--use_llama3_1", action="store_true", help="use_llama3_1")
    parser.add_argument('--instruct_pssg', type=str, default="original") # original, filter_irrelevant, summary, filter_irrelevant_summary
    parser.add_argument("--ctx_original_qs", action="store_true", help="contain original qs within ctx")

    # longlora: args    
    # parser.add_argument('--material', type=str, default="")
    # parser.add_argument('--question', type=str, default="")
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--sft_model', type=str, default="")
    parser.add_argument('--cache_dir', type=str, default=None) # ./cache
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--temperature', type=float, default=0.6, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--max_gen_len', type=int, default=512, help='')
    args = parser.parse_args()
    ####################

    lines = jload(args.input_data)

        
    
    if args.use_llama3_1:
        respond, tokenizer = build_lamma(args)
    else:
        respond = build_lamma(args)
        tokenizer = None

    out_path = args.output_path if args.output_path is not None else f'{split}_chatgpt_ZSL_...jsonl'
    output_root = args.output_root if args.output_root is not None else root
    with jsonlines.open(os.path.join(output_root, out_path), mode='a') as writer:
        for line in tqdm(lines):
            
            new_line = {'question': line['output']}
            new_line['answer'] = generate_rewrite(line, respond, args, tokenizer=tokenizer)
            

            writer.write(new_line)
    
