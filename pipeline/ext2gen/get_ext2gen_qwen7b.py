from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch
from ..common import ext2gen_qwen7b_path


class Qwen7bGenerator:
    
    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path or ext2gen_qwen7b_path
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
    
    def load(self):
        if self._is_loaded:
            return
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self.checkpoint_path, 
            torch_dtype="auto", 
            device_map="auto"
        )
        
        self._is_loaded = True
    
    def generate(self, formatted_prompt):
        if not self._is_loaded:
            self.load()
        
        messages = [
            {"role": "system", "content": "당신은 과학기술정보 전문가입니다. 한국어에 능통합니다. CONTEXT에 기반하여 사용자에게 질문에 답변하세요."},
            {"role": "user", "content": formatted_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=2048,
            do_sample=False,
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


# 전역 싱글톤
_global_generator = None

def get_generator():
    global _global_generator
    if _global_generator is None:
        _global_generator = Qwen7bGenerator()
    return _global_generator


def generate_qwen7b_dpo(formatted_prompt):
    generator = get_generator()
    return generator.generate(formatted_prompt)