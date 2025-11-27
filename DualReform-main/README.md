# LEARNING CONVERSATIONAL QUERY REWRITING WITHOUT SUPERVISED REFERENCE PASSAGES

## About

- The source code of the paper **LEARNING CONVERSATIONAL QUERY REWRITING
WITHOUT SUPERVISED REFERENCE PASSAGES**.
- The source code is mainly based on the pytorch implementations for conversational query reformulation and finetuning LLMs: [ConvGQR](https://github.com/fengranMark/ConvGQR), [LLM-IQR](https://github.com/smartyfh/InfoCQR), [LongLora](https://github.com/dvlab-research/LongLoRA)

## Environment

The system I used and tested in

- Ubuntu 20.04.4 LTS
- Python 3.10.4
- pytorch==2.1.2
- transformers 4.43.0
- NVIDIA RTX A6000 Ada GPUs
- Faiss-gpu
- Pyserini
- sentence-transformers

## Usage

Please follow the instructions to train our model:

1. Data Preparation
2. Generate Candidate Query Rewrites
3. Generate Pseudo Reference
4. Preference Optimization
5. Repeat step 3. and 4.

### 1. Data Preparation

- QReCC and TopiOCQ

  Two public datasets can be downloaded from [QReCC](https://github.com/apple/ml-qrecc), and [TopiOCQA](https://github.com/McGill-NLP/topiocqa).
- SciConvQA
  Seed journal dataset for generating SciConvQA can be downladed on the https://aida.kisti.re.kr/data/b22c73ed-fa19-47b0-87b3-a509df8380e5

  For the processing and generation of SciConvQA: please refer to `CQR/sciconvqa_generation`folder. Following the template, you can also generate your own conversational datasets.
- Data preprocessing for building databases and retrieval systems (sparse and dense retrievers)
  Please refer to `CQR/db_build` folder

### 2. Generate Candidate Query Rewrites

- We employ GPT3.5-turbo-0125 to generate candidate query rewrites; please refer to `CQR/gpt_rewrite_candidate_queries.py`
- We employ llama3.1-8b-inst to generate candidate query expansion; please refer to `CQR/query_expansion_llama3.py`

### 3. Generate Pseudo Reference Passages

- We refine response, and retrieve pseudo reference passages from database

  - Regarding refining response, please refer to `CQR/refine_response.py`

    ```
    CUDA_VISIBLE_DEVICES=$gpu_id python3 refine_response.py \
      --input_data train_{topiocqa or sciconvqa}.json \
      --root /your/path/ \
      --prompt_type ${p_type} \
      --instruct_pssg $inst_pssg \
      --output_root /your/output/path \
      --output_path train_refined_response.jsonl \
      --base_model /your/output/to/adapter/path \
      --sft_model /your/output/to/sft-model/path \
      --context_size 4096 \
      --max_gen_len 512 \
      --use_llama3_1 \
      --ctx_original_qs \
      --selfask_prompt \
      --selfask_prompt_txt $formatted_prompt_txt \
      --add_real_answer \
      --flash_attn True 
    ```
  - Regarding retrieving pseudo reference passages, please refer to please refer to `CQR/sparse_retrieve/bm25` folder

    ```
    python bm25/bm25_on_{topiocqa or sciconvqa}.py \
    --input_query_path  /your/path/${fname} \
    --gold_qrel_file_path /your/path/train_gold.trec \
    --output_dir_path /your/output/path \
    --output_f train_pseudo_refs_${fname}.trec \
    --query_type oracle
    ```
* Generate Preference Feedback
  please refer to `CQR/preference_generation/*.ipynb`
  - create_pref_data_*.ipynb: creates preference data for DPO optimization
  - create_sft_*.ipynb: creates data for SFT optimization

    
### 4. Preference Optimization


* Optimization
  please refer to `PreferenceOptimization` folder
  * for SFT, please refer to `PreferenceOptimization/supervised-fine-tune-qlora_llama3.py`

    ```
    CUDA_VISIBLE_DEVICES=$gpu_id accelerate launch --num_processes 1 --main_process_port $port_id supervised-fine-tune-qlora_llama3.py  \
        --model_name_or_path "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --data_path /your/path/{Topiocqa or SciConvQA}_SFT.json \
        --output_dir_path /your/output/path/sft_model \
        --output_f /your/output/fname \
        --epochs 1 \
        --lr $lr 
    ```
  * for DPO, please refer to `PreferenceOptimization/dpo_llama3_accelerate.py`

    ```
    CUDA_VISIBLE_DEVICES=$gpu_id accelerate launch --num_processes 1 --main_process_port $port_id dpo_llama3_accelerate.py  \
        --lr $lr \
        --beta $beta \
        --dataset_path /your/path/pref_data_{Topiocqa or SciConvQA}_init.csv \
        --data {Topiocqa or SciConvQA} \
        --sft_model /your/output/to/sft-model/path \
        --train_val_split 0.05 \
        --output_dir /your/output/path/1R_dpo_model \
        --wandb_name "1R_dpo" \
        --max_length 4096 \
        --per_device_bs 1 \
        --eval_per_device_bs 2 \
        --per_epoch_steps 125 \
        --grad_acc_steps 32 \
        --train_log_period 10 \
        --eval_log_period 100 
    ```

Please follow the instructions to inference our model:

- for query reformulation, please refer to `CQR/reformulate_query.py`

  ```
  CUDA_VISIBLE_DEVICES=$gpu_id python3 reformulate_query.py \
      --input_data dev_new.json \
      --root /your/path/to/{Topiocqa or SciConvQA} \
      --prompt_type oracle \
      --instruct_pssg icl \
      --output_root /your/output/path \
      --output_path test_{fname}.jsonl \
      --base_model /your/output/to/adapter/path \
      --sft_model /your/output/to/sft-model/path \
      --context_size 4096 \
      --max_gen_len 512 \
      --use_llama3_1 \
      --ctx_original_qs
  ```

Please follow the instructions to evaluate our model:

- for retrieval evaluation, please refer to `CQR/sparse_retrieve` and `CQR/dense_retrieve`
  ```
  # sparse retriever
  python bm25/bm25_on_{topiocqa or sciconvqa}.py \
    --input_query_path  /your/path/${fname} \
    --gold_qrel_file_path /your/path/dev_gold.trec \
    --output_dir_path /your/output/path \
    --output_f test_pseudo_refs_${fname}.trec \
    --query_type oracle

  # dense retriever
  CUDA_VISIBLE_DEVICES=$gpu_id python3 cs-shortcut/run_dense_search_{topiocqa or sciconvqA}.py \
      --preprocessed_data_path /your/output/path \
      --dense_index_path /your/path/{topiocqa or sciConvQA}/dense_index_gtr_large \
      --model_name_or_path sentence-transformers/gtr-t5-large \
      --test_file_path /your/output/path/{topiocqa or sciConvQA}/${fname} \
      --eval_type oracle \
      --output_f ${fname}_oracle.trec 

  ```
- for generation evaluation, please refer to `PreferenceOptimization/generation_eval.ipynb`

# License

This repository is released under the Apache 2.0 license.
