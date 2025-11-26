import os, json, argparse
from typing import List, Dict, Any
from pathlib import Path
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..common import model_name, device

def load_value_head(weight_path: str, hidden_size: int, device: torch.device):
    ''' slp to classify need_retrieval/no_retrieval '''
    sd = torch.load(weight_path, map_location="cpu")
    head = nn.Linear(hidden_size, 2, bias=False)
    head.weight.data = sd["score.weight"]
    return head.to(device).eval()

def tokenize_batch(tokenizer, texts: List[str], device, max_len=2048):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    for k in enc:
        enc[k] = enc[k].to(device)
    return enc

@torch.inference_mode()
def qwen_cls_logits_last_token(
    backbone: AutoModelForCausalLM,
    head: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor
):
    """
    - last_hidden_state: [B, T, H]
    - 마지막 유효 토큰 인덱스: attention_mask 합 - 1
    - 해당 위치의 토큰 벡터에 head를 적용 → [B, 2] 로짓
    """
    core = getattr(backbone, "model", getattr(backbone, "base_model", backbone))

    outputs = core(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        use_cache=False,
    )
    if hasattr(outputs, "last_hidden_state"):
        hidden_states = outputs.last_hidden_state  # [B, T, H]
    else:
        hidden_states = outputs[0]

    '''
    last_idx = attention_mask.sum(dim=1) - 1  # [B]
    b_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    last_vec = hidden_states[b_idx, last_idx]  # [B, H]
    '''

    # ensure attention_mask / last_idx are on same device as hidden_states
    last_idx = (attention_mask.to(torch.long).to(hidden_states.device).sum(dim=1) - 1).clamp(min=0)  # [B], int64 on correct device
    b_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    last_vec = hidden_states[b_idx, last_idx]  # [B, H]

    # move last_vec to head's device before applying head (prevent device-mismatch)
    head_device = next(head.parameters()).device if any(p is not None for p in head.parameters()) else hidden_states.device
    if last_vec.device != head_device:
        last_vec = last_vec.to(head_device)

    logits = head(last_vec)  # [B, 2]
    return logits

@torch.inference_mode()
def koni_cls_logits_last_token(tokenizer: AutoTokenizer,
    backbone: AutoModelForCausalLM,
    head: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor
):
    """
    - last_hidden_state: [B, T, H]
    - 마지막 유효 토큰 인덱스: attention_mask 합 - 1
    - 해당 위치의 토큰 벡터에 head를 적용 → [B, 2] 로짓
    """
    eos_token = tokenizer.eos_token_id
    eot_token = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    core = getattr(backbone, "model", getattr(backbone, "base_model", backbone))

    outputs = core(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False, eos_token_id = [eos_token, eot_token]
    )
    hidden_states = outputs.hidden_states[-1]
    last_idx = attention_mask.sum(dim=1) - 1  # [B]
    b_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    last_vec = hidden_states[b_idx, last_idx]  # [B, H]

    logits = head(last_vec)  # [B, 2]
    return logits

# classifier 존재하는지 확인 (adaptive_classifier/koni, qwen) (self_head_path 존재하는지 확인)
# classifier 존재한다면 inference 통해 해당 질문이 retrieval 필요한지 아닌지 판단
# 존재하지 않는다면 classifier train code로 넘어감 (classifier train code에선 idk dataset 존재하는지 확인, 존재하지 않는다면 idk create code로 넘어감)

def adaptive_classify(query):
    self_head_path = (Path(__file__).resolve().parent / "adaptive_classifier" / model_name.split("/")[0] / "self_cls.pth")
    threshold = 0.1
    
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if model_name == "Qwen/Qwen2.5-7B-Instruct":
        # check whether qwen_slp classifier file exists in adaptive_classifier/qwen
        last_token = qwen_cls_logits_last_token
    elif model_name == "KISTI-KONI/KONI-4B-instruct-20250901":
        # check whether koni_slp classifier file exists in adaptive_classifier/koni
        last_token = koni_cls_logits_last_token
    else:
        assert model_name in ["Qwen/Qwen2.5-7B-Instruct", "KISTI-KONI/KONI-4B-instruct-20250901"]
        # error! not appropriate model name

    backbone = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float,
        device_map="auto" if torch.cuda.device_count() > 0 else None,
        low_cpu_mem_usage=True,
    )
    backbone.eval()
    head = load_value_head(self_head_path, backbone.config.hidden_size, device)

    with torch.inference_mode():
        batch_q = [query]
        enc = tokenize_batch(tok, batch_q, device) # koni_cls_logits_last_token()
        
        logits = last_token( 
            backbone=backbone,
            head=head,
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        probs = softmax(logits, dim=1)
        
        for _, p in zip(batch_q, probs):
            self_prob = float(p[0].item())
            need_retrieve = bool(self_prob >= threshold)

    return need_retrieve

if __name__=="__main__":
    query = "What is the impact of climate change on marine biodiversity?"
    need_retrieval = adaptive_classify(query)
    print(f"Query: {query}")
    print(f"Need retrieval: {need_retrieval}")
    
    