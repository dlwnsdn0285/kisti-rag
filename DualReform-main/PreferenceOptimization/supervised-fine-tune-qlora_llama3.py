import os

# Set the cache directory before importing transformers
os.environ["TRANSFORMERS_CACHE"] = "/data/../llm_cache"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# os.environ["LD_LIBRARY_PATH"]="/home/../anaconda3/envs/longlora/lib/python3.10/site-packages/nvidia/cusparse/lib/libcusparse.so.12"
import io
import json
import argparse

from datasets import Dataset
from datasets import DatasetDict
from accelerate import Accelerator

from huggingface_hub import login
import re
import random
from multiprocessing import cpu_count

from transformers import BitsAndBytesConfig
import torch
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from transformers import TrainingArguments
from transformers import AutoTokenizer
import transformers

login("<YOUR_HF_TOKEN>")

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

def get_dataset(args):
    list_data_dict = jload(args.data_path)
    for dt in list_data_dict:
        new_list = [{'content':dt['instruction'], 'role':'user'},
                {'content':dt['output'], 'role':'assistant'},]
        dt['concat'] = new_list

    # Convert the list of dictionaries to a Hugging Face Dataset
    hf_dataset = Dataset.from_list(list_data_dict)
    # hf_dataset

    dataset_dict = {"train": hf_dataset,
                    # "test": raw_datasets["test_sft"].select(indices)
                }

    raw_datasets = DatasetDict(dataset_dict)
    return raw_datasets

def apply_chat_template(example, tokenizer):
    messages = example["concat"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example

def train():
    args = get_args()
    raw_datasets = get_dataset(args)

    # model_id = "mistralai/Mistral-7B-v0.2"
    model_id = args.model_name_or_path # "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]



    column_names = list(raw_datasets["train"].features)
    raw_datasets = raw_datasets.map(apply_chat_template,
                                    num_proc=cpu_count(),
                                    fn_kwargs={"tokenizer": tokenizer},
                                    remove_columns=column_names,
                                    desc="Applying chat template",)

    # set pad_token_id equal to the eos_token_id if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048


    # create the splits
    train_dataset = raw_datasets["train"]
    # eval_dataset = raw_datasets["test"]

    # for index in random.sample(range(len(raw_datasets["train"])), 1):
    #   print(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")
    #   print("#####################################")

    # specify how to quantize the model
    quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
    )
    device_map = {"": Accelerator().local_process_index} # "auto" #{"": torch.cuda.current_device()} if torch.cuda.is_available() else None

    model_kwargs = dict(
    #     attn_implementation=False,#"flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
        torch_dtype="auto",
        use_cache=False, # set to False as we're going to use gradient checkpointing
        device_map=device_map,
        quantization_config=quantization_config,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs
    )
    
    initial_token_count = len(tokenizer)
    response_template = "Rewrite:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    # "Rewrite: Then what happens after the layer closer to the top of the container is poured off with decantation?"
    # get_response_template(prompt_format) # '### Response:\n'
    added_token_count = tokenizer.add_special_tokens({"additional_special_tokens": [response_template]})
    # added_token_count
    model.resize_token_embeddings(new_num_tokens=initial_token_count+added_token_count)

    data_collator= DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)



    # path where the Trainer will save its checkpoints and logs
    trained_model_id = args.output_f # "Llama-3.1-8B-sft-lora-ultrachat"
    output_dir = args.output_dir_path + trained_model_id # 'kaggle/working/' + trained_model_id
    
    steps_epoch = len(train_dataset)//64
    max_steps = steps_epoch * args.epochs
    save_steps = (steps_epoch * args.epochs) // args.n_save

    # based on config
    training_args = TrainingArguments(
        fp16=False, # specify bf16=True instead when training on GPUs that support bf16 else fp16
        bf16=True,
        max_steps=max_steps,
        gradient_accumulation_steps=32,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,#2e-5,
        # log_level="info",
        logging_steps=1,
        # logging_strategy="steps",
        lr_scheduler_type="constant_with_warmup",
        warmup_steps =20,
        output_dir=output_dir,
        overwrite_output_dir=True,
        # per_device_eval_batch_size=1, # originally set to 8
        per_device_train_batch_size=2, # originally set to 8
        # push_to_hub=True,
        hub_model_id=trained_model_id,
        # hub_strategy="every_save",
        # report_to="tensorboard",
        report_to="none",
        save_strategy="steps",
        save_steps = save_steps,
        save_total_limit=args.n_save,
        seed=42,
    )

    peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # del trainer
    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        # model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        # packing=True,
        # eval_dataset=eval_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_seq_length=tokenizer.model_max_length,
        # packing=False
        )

    train_result = trainer.train()



def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--data_path", type=str, default="/data/../nlp_data/LongAlpaca-12k/Best_12k.json")
    parser.add_argument("--output_dir_path", type=str, default="/data2/../nlp_data/qrecc_llama3_1/")
    parser.add_argument("--output_f", type=str, default="Llama-3.1-8B-sft-kisti-QE-llama3-best10k")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--n_save", type=int, default=2)

    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')


    args = parser.parse_args()
    print(args)
    return args

if __name__ == "__main__":
    train()
