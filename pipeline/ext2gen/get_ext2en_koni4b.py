from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch
import transformers
from ..common import ext2gen_koni4b_path


class Koni4bGenerator:
    
    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path or ext2gen_koni4b_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._is_loaded = False
    
    def load(self):
        if self._is_loaded:
            return
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self.checkpoint_path, 
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager"
        )
        self.model.eval()
        
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        self._is_loaded = True
    
    def generate(self, formatted_prompt):
        if not self._is_loaded:
            self.load()
        
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
        
        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        ]
        
        outputs = self.pipeline(
            prompt,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=False,
            temperature=0.0,
            top_p=None,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        return outputs[0]["generated_text"][len(prompt):]


# 전역 싱글톤
_global_generator = None

def get_generator():
    global _global_generator
    if _global_generator is None:
        _global_generator = Koni4bGenerator()
    return _global_generator


def generate_koni4b_dpo(formatted_prompt):
    generator = get_generator()
    return generator.generate(formatted_prompt)