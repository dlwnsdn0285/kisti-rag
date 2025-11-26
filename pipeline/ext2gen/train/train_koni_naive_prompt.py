import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer
import json, random
import multiprocessing as mp
import os
from accelerate import Accelerator
from transformers.trainer_utils import get_last_checkpoint

# CUDA_VISIBLE_DEVICES=1,2,3,6 torchrun --nproc_per_node=4 train.py
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py

import os


def main():

    def format_prompt(query: str, retrieved_documents: str) -> str:
        PROMPT = f"""
    Answer the QUERY below. Refer to the provided CONTEXT if needed.
    QUERY: {query}
    CONTEXT: {retrieved_documents}
    Instructions:
    - Answer only the query directly and naturally
    - Do not evaluate or analyze the context
    - Do not mention the context, documents, or sources in your response
    - Response must be in JSON format only
    JSON format:
    {{"Answer": "your direct answer here"}}
    """
        return PROMPT

    acc = Accelerator()
    local_rank = acc.local_process_index  # 또는 int(os.environ.get("LOCAL_RANK", 0))

    device_map = {"": local_rank}
    
    # 1. 모델 및 토크나이저 로드
    model_name = 'KISTI-KONI/KONI-4B-instruct-20250901'

    # QLoRA를 위한 4bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        #attn_implementation="flash_attention_2", 
    )
    model.config.use_cache = False

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. LoRA 설정
    lora_config = LoraConfig(
        r=4,  # LoRA rank
        lora_alpha=8,  # LoRA alpha
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 모델을 kbit training에 맞게 준비
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 데이터셋 준비
    with open("/data/minjeong/KISTI2025/ext2gen/train_data/dpo_data/naive_prompt/dpo_data_naive_prompt.json", "r", encoding="utf-8") as f:
        dpo_data = json.load(f)
    with open(f'/data/minjeong/KISTI2025/kisti-rag/data/train/sample/queries.json', 'r', encoding='utf-8') as f:
        queries = json.load(f)
    with open(f'/data/minjeong/KISTI2025/ext2gen/train_data/train_data.json', 'r', encoding='utf-8') as f:
        retrieved_corpus = json.load(f)
        
    total_data = []
    for q_id in list(dpo_data.keys()):
        input_chunk = retrieved_corpus[q_id]["relevant_chunk"] + retrieved_corpus[q_id]["irrelevant_chunk"]
        random.shuffle(input_chunk)
        prompt = format_prompt(queries[q_id]['query'], '\n\n'.join(input_chunk))
        for data in dpo_data[q_id]:
            total_data.append(
                {
                    "prompt": prompt,
                    "chosen": data["chosen"],
                    "rejected": data["rejected"]
                }
            )

    print(f'dpo preference train data sieze: {len(total_data)}')

    dataset = Dataset.from_list(total_data)

    # 4. DPO 학습 설정
    training_args = DPOConfig(
        output_dir="./koni4b_dpo_output_total_naive_prompt",
        per_device_train_batch_size=2,  # GPU 메모리에 따라 4로 증가 가능
        gradient_accumulation_steps=16,  # 배치 크기 증가시 4로 감소 가능
        num_train_epochs=2,
        learning_rate=5e-5,
        fp16=False,
        bf16=True,
        logging_steps=100,
        save_steps=500,  # 저장 빈도 줄이기
        save_total_limit=2,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        report_to="wandb",  # 학습 모니터링
        remove_unused_columns=False,
        dataloader_num_workers=0,  # 데이터 로딩 속도 향상
        dataloader_pin_memory=False,
        # DPO 특정 파라미터
        beta=0.1,  # DPO loss의 beta 값
        max_length=8192,  # 최대 시퀀스 길이
        max_prompt_length=7000,  # 최대 프롬프트 길이 - 주로 4000대임..
        ddp_find_unused_parameters=False,  # 이미 권장
    )

    def clear_gpu_memory():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()

    # 5. 모델 메모리 최적화 설정
    # Activation checkpointing 활성화
    model.gradient_checkpointing_enable()

    # 6. 메모리 정리
    clear_gpu_memory()

    # 5. DPO Trainer 초기화 및 학습
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # QLoRA에서는 reference model을 None으로 설정
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # 체크포인트에서 자동 복구
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            print(f"Resuming from checkpoint: {last_checkpoint}")

    
    # 학습 시작
    print("학습을 시작합니다...")
    dpo_trainer.train(resume_from_checkpoint=last_checkpoint)

    # 6. 모델 저장
    print("모델을 저장합니다...")
    dpo_trainer.save_model("./koni4b_dpo_final_total_naive_prompt")
    tokenizer.save_pretrained("./koni4b_dpo_final_total_naive_prompt")

    print("학습 완료!")

    # 7. 추론 테스트 (선택사항)
    model.eval()
    test_prompt = "배에 배검은별무늬병원균이 잎에 감염되는 경우에 대해 설명해주세요."
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,
        )

    print("\n=== 추론 결과 ===")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    

if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    mp.freeze_support()
    main()