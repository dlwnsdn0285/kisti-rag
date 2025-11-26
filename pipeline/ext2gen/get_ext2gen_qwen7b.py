from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from ..common import ext2gen_qwen7b_path

checkpoint_path = ext2gen_qwen7b_path

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype="auto", device_map="auto").cuda()

def generate_qwen7b_dpo(formatted_prompt, tokenizer=tokenizer, model=model):
    messages = [
        {"role": "system", "content": "당신은 과학기술정보 전문가입니다. 한국어에 능통합니다. CONTEXT에 기반하여 사용자에게 질문에 답변하세요."},
        {"role": "user", "content": formatted_prompt}  # Pass the formatted prompt string here
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048,
        do_sample=False,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
