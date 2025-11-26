from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch
import transformers
from ..common import ext2gen_koni4b_path


checkpoint_path = ext2gen_koni4b_path

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoPeftModelForCausalLM.from_pretrained(
    checkpoint_path, 
    torch_dtype=torch.bfloat16,  # 또는 torch.float16
    device_map="auto",
    attn_implementation="eager" 
)

model.eval()

# 파이프라인 생성
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

def generate_koni4b_dpo(formatted_prompt):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "당신은 과학기술정보 전문가입니다. 한국어에 능통합니다. CONTEXT에 기반하여 사용자에게 질문에 답변하세요."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": formatted_prompt}]
        }
    ]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.0,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id
    )

    return outputs[0]["generated_text"][len(prompt):]

