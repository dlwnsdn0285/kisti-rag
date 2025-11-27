import os
import sys
import math
import torch
import numpy as np
import pandas as pd
import argparse
import textwrap
import transformers
from peft import PeftModel
from transformers import GenerationConfig, TextStreamer, BitsAndBytesConfig
from llama_attn_replace import replace_llama_attn


from dataclasses import dataclass, field
from typing import Dict, Optional

from accelerate import Accelerator
import datasets
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed
from transformers import TextStreamer

from trl import DPOConfig, DPOTrainer

import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from llama_attn_replace_sft import replace_llama_attn
from gptneox_attn_replace import replace_gpt_neox_attn
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

# import os
# os.environ['CUDA_HOME'] = '/home/../anaconda3/envs/selfrag/'
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

os.environ["TRANSFORMERS_CACHE"] = "/data/../llm_cache"
from huggingface_hub import login

login("<YOUR_HF_TOKEN>")
import argparse

def apply_chat_template_content(example, tokenizer):
    # messages = example # ["concat"][:1]
    messages = [{"role": "system", "content": ""}, {'content':example, 'role':'user'}]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    add = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    # example["text"] = 
    return tokenizer.apply_chat_template(messages, tokenize=False)+add

def add_template_response(example,):
    add = "<|eot_id|>\n"
    return example+add

def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
                "prompt": [question for question in samples["template_question"]],
                "chosen": samples["template_response_j"],
                "rejected": samples["template_response_k"],
            }


def main():
    parser = argparse.ArgumentParser(description='Train a DPO model with customizable parameters.')
    
    # Add arguments: per_device_bs
                    # eval_per_device_bs
                    # per_epoch_steps
                    # grad_acc_steps
                    # data
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1, help='dpo beta')
    parser.add_argument('--dataset_path', type=str, default='/data/../nlp_data/LongAlpaca-12k/pref_data.csv', help='Path to the dataset')
    parser.add_argument('--data', type=str, default='qrecc', help='dataset')

    parser.add_argument('--sft_model', type=str, default="/data2/../llm_cache/sft/qrecc_llama3_1-Best_13k-checkpoint-570", help='Path to the model')
    parser.add_argument('--ref_model', type=str, default=None, help='Path to the model')
    parser.add_argument('--wandb_name', type=str, default="", help='wandb')
    parser.add_argument('--train_val_split', type=float, default=0.05, help='Train-validation split ratio')
    
    parser.add_argument('--output_dir', type=str, default='/data2/../nlp_data/qrecc_longlora/llama3_1_dpo_bs32_lr2e-4', help='Output directory')
    parser.add_argument('--max_length', type=int, default=4096, help='max_length')
    parser.add_argument('--max_target_length', type=int, default=None, help='max_target_length')
    parser.add_argument('--per_device_bs', type=int, default=3, help='Batch size per device')
    parser.add_argument('--eval_per_device_bs', type=int, default=7, help='Evaluation batch size per device')
    parser.add_argument('--per_epoch_steps', type=int, default=562, help='Number of training steps per epoch')
    parser.add_argument('--grad_acc_steps', type=int, default=11, help='Number of grad accumulations')
    parser.add_argument('--n_save', type=int, default=2, help='n_save of checkpoints')
    parser.add_argument('--train_log_period', type=int, default=10, help='train_log_period')
    parser.add_argument('--eval_log_period', type=int, default=100, help='eval_log_period')
    args = parser.parse_args()
    # if args.data == "qrecc":
        
    print(args.per_device_bs * args.grad_acc_steps)
    total_bs = args.per_device_bs * args.grad_acc_steps
    loading_args_dict = {
        'base_model': args.sft_model,
        'cache_dir': None,
        'flash_attn': True,
        'context_size': args.max_length,
        'sanity_check': False,
        'max_length':args.max_length,
        "model_dtype":"bfloat16",
        # "trainable_params":"embed,norm",
    }
    # Creating an args object
    loading_args = argparse.Namespace(**loading_args_dict)
    # Accessing arguments
    print(loading_args)

    #! 1. load model
    config = transformers.AutoConfig.from_pretrained(
        loading_args.base_model,
        cache_dir=loading_args.cache_dir,
    )
    config.use_cache = False

    # specify how to quantize the model
    quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
    )

    device_map = {"": Accelerator().local_process_index} # "auto" #{"": torch.cuda.current_device()} if torch.cuda.is_available() else None
    model_kwargs = dict(
        # attn_implementation=False,#"flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16, # "auto", # torch.bfloat16
        # use_cache=False, # set to False as we're going to use gradient checkpointing
        device_map=device_map,
        quantization_config=quantization_config,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
            loading_args.base_model,
            config=config,
            **model_kwargs
        )


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        loading_args.base_model,
    )

    model.resize_token_embeddings(len(tokenizer))
    
    if args.ref_model is not None:
        config_ref = transformers.AutoConfig.from_pretrained(
                args.ref_model,
                cache_dir=loading_args.cache_dir,
                )
        config_ref.use_cache = False
        model_ref_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16, # "auto", # torch.bfloat16
        device_map=device_map,
        # quantization_config=quantization_config,
                )
        model_ref = transformers.AutoModelForCausalLM.from_pretrained(
            args.ref_model,
            config=config_ref,
            **model_ref_kwargs
        )
        model_ref.resize_token_embeddings(len(tokenizer))
        model_ref.eval()
    else: 
        model_ref = None
        

    # #! test generation
    # model.eval()
    # input_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven a question, its previous questions (Q) and answers (A), decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the context.\n\nContext: [Q: When was Born to Fly released? A: Sara Evans's third studio album, Born to Fly, was released on October 10, 2000.]\nQuestion: Was Born to Fly well received by critics?\nRewrite: Was Born to Fly well received by critics?\n\nContext: [Q: When was Keith Carradine born? A: Keith Ian Carradine was born August 8, 1949. Q: Is he married? A: Keith Carradine married Sandra Will on February 6, 1982.]\nQuestion: Do they have any children?\nRewrite: Do Keith Carradine and Sandra Will have any children?\n\nContext: [Q: Who proposed that atoms are the basic units of matter? A: John Dalton proposed that each chemical element is composed of atoms of a single, unique type, and they can combine to form more complex structures called chemical compounds.]\nQuestion: How did the proposal come about?\nRewrite: How did John Dalton's proposal that each chemical element is composed of atoms of a single unique type, and they can combine to form more complex structures called chemical compounds come about?\n\nContext: [Q: What is it called when two liquids separate? A: Decantation is a process for the separation of mixtures of immiscible liquids or of a liquid and a solid mixture such as a suspension. Q: How does the separation occur? A: The layer closer to the top of the container-the less dense of the two liquids, or the liquid from which the precipitate or sediment has settled out-is poured off.]\nQuestion: Then what happens?\nRewrite: Then what happens after the layer closer to the top of the container is poured off with decantation?\n\nContext: [Q: What was Faith Hill's first country album? A: Take Me as I Am is the debut studio album by country singer Faith Hill. Q: What was a single from the album? A: The first single from Faith Hill's Take Me as I am is Wild One. Q: Was the song a success? A: Hill's rendition was also her first Number One, spending the first four chart weeks of 1994 at the top of the Billboard Hot Country Singles & Tracks chart. Q: Did the album perform well? A: Take Me as I am has been certified 3× platinum in the United States for shipments of three million copies. Q: Did she write her own songs? A: Faith Hill performs songs other people wrote. Q: Did she tour? A: Faith Hill's Soul2Soul II Tour 2006 with McGraw became the highest-grossing country tour of all time. Q: Who did she tour with? A: Faith Hill toured with Tim Mcgraw in 2006.]\nQuestion: Did they sing together?\nRewrite:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    # # print(input_text)

    # inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    # streamer = TextStreamer(tokenizer)

    # # Generate text
    # print("test generation")
    # outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.6,
    #             top_p=0.9, use_cache=True, streamer=streamer)
    # generated_text = tokenizer.decode(outputs[0],) # skip_special_tokens=True)


    # # 2. Load the dataset

    # df = pd.read_csv('/data/../nlp_data/LongAlpaca-12k/pref_data_topiocqa.csv')
    df = pd.read_csv(args.dataset_path)
    # df.head()

    df['template_question'] = df['question'].apply(lambda example: apply_chat_template_content(example, 
                                                                                               tokenizer))
    df['template_response_j'] = df['response_j'].apply(lambda example: add_template_response(example))
    df['template_response_k'] = df['response_k'].apply(lambda example: add_template_response(example))

    df_eval = df.sample(frac=args.train_val_split, replace=False, random_state=1)
    df_train = df.drop(df_eval.index).reset_index()
    df_eval = df_eval.reset_index()
    # df_eval.head()

    df_train_hf = datasets.Dataset.from_pandas(df_train)
    df_eval_hf = datasets.Dataset.from_pandas(df_eval)
    
    df_train_hf = df_train_hf.map(
            return_prompt_and_responses,
            batched=True,
            num_proc=24,
            remove_columns=df_train_hf.column_names,
        )
    df_train_hf = df_train_hf.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= 1000000000 # 2000 # 1000000000
        and len(x["prompt"]) + len(x["rejected"]) <= 1000000000 # 2000 # 1000000000
    )

    # df_train_hf

    df_eval_hf = df_eval_hf.map(
            return_prompt_and_responses,
            batched=True,
            num_proc=24,
            remove_columns=df_eval_hf.column_names,
        )
    df_eval_hf = df_eval_hf.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= 1000000000 # 2000 # 1000000000
        and len(x["prompt"]) + len(x["rejected"]) <= 1000000000 # 2000 # 1000000000
    )

    # df_eval_hf

    train_dataset = df_train_hf
    eval_dataset= df_eval_hf

    max_steps = args.per_epoch_steps * 1
    save_steps = max_steps//args.n_save
    

    lr_str = f"{args.lr:.0e}"
    # print(a_str)
    # 4. initialize training arguments:
    training_args = DPOConfig(
            per_device_train_batch_size=args.per_device_bs,
            per_device_eval_batch_size=args.eval_per_device_bs,
            max_steps=max_steps, # 5 epochs
            logging_steps=args.train_log_period,
            save_steps=save_steps, # max_steps//args.n_save
            save_total_limit=args.n_save, # args.n_save
            gradient_accumulation_steps=args.grad_acc_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=args.lr,
            eval_strategy="steps",
            eval_steps=args.eval_log_period,
            output_dir=args.output_dir, #"/data2/../nlp_data/qrecc_longlora/llama3_1_dpo_bs32_lr2e-4",
            report_to="wandb",
            lr_scheduler_type="cosine",
            warmup_steps=10,
            optim="adamw_torch", #"adamw_torch",
            bf16=True,
            remove_unused_columns=False,
            run_name=f"dpo_llama3_1_bs{total_bs}_{lr_str}_{args.data}_{args.wandb_name}" \
              if args.wandb_name else f"dpo_llama3_1_bs{total_bs}_{lr_str}_{args.data}",
            seed=0,
        )
    
    peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "out_proj",
                "fc_in",
                "fc_out",
                "wte",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )

    model.config.use_cache = False         # required for gradient checkpointing
    model = get_peft_model(model, peft_config)

        
    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
            model,
            ref_model=model_ref, #args.ref_model, # model_ref, # if no ref_model, then None is input
            args=training_args,
            beta=args.beta, # default: 0.1,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            # peft_config=peft_config,
            max_prompt_length=args.max_length,
            max_length=args.max_length, 
            max_target_length=args.max_target_length
        )

    # 6. train
    dpo_trainer.train()
    # dpo_trainer.save_model("/data2/../nlp_data/qrecc_longlora/llama3_1_dpo_bs32_lr2e-4")

#########################################################


if __name__ == "__main__":
    main()
