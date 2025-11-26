from get_ext2en_koni4b import generate_koni4b_dpo # when use ext2gen
from get_ext2gen_qwen7b import generate_qwen7b_dpo # when use ext2gen
from ..common import model_name


def ext2gen_model():
    if model_name == "Qwen/Qwen2.5-7B-Instruct":
        generate_ext2gen = generate_qwen7b_dpo
    elif model_name == "KISTI-KONI/KONI-4B-instruct-20250901":
        generate_ext2gen = generate_koni4b_dpo
    else:
        assert model_name in ["Qwen/Qwen2.5-7B-Instruct", "KISTI-KONI/KONI-4B-instruct-20250901"]
        
    return generate_ext2gen