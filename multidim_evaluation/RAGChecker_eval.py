import os
import sys
import warnings

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore', message='.*generation flags.*')

from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
from evaluation.eval import *
import json

if len(sys.argv) < 2:
    print("Usage: python evaluate_rag.py <input_path>")
    sys.exit(1)

input_path = sys.argv[1]

input_filename = os.path.basename(input_path)  
filename_without_ext = input_filename.replace('_formatted.json', '')  
output_filename = f"{filename_without_ext}_Eval_Result.json"
output_path = f'./ragchecker_result/{output_filename}'

os.makedirs('./ragchecker_result/temp', exist_ok=True)

print(f"Input path: {input_path}")
print(f"Output path: {output_path}")

final_results = {}
model_results = {}
temp_output_path = f'./ragchecker_result/temp/temp.json'

evaluate_by_dicts(input_path, temp_output_path)

with open(input_path) as fp:
    ragchecker_results = RAGResults.from_json(fp.read())

evaluator = RAGChecker(
    batch_size_extractor=32,
    batch_size_checker=32,
)

evaluator.evaluate(ragchecker_results, all_metrics)
ragchecker_json = json.loads(ragchecker_results.to_json())
model_results['ragchecker'] = ragchecker_json['metrics']

from refchecker.llm_module.qwen_model import unload_qwen_model
unload_qwen_model()

from evaluation.llm_eval import *
llm_eval_score = RAG_eval_w_LLM(input_path, llm_model='Qwen/Qwen3-30B-A3B-Instruct-2507')
model_results['binary_llm_eval'] = {"final_avg_score": llm_eval_score}

final_results["Result"] = model_results

if os.path.exists(temp_output_path):
    with open(temp_output_path, 'r', encoding='utf-8') as temp_reader:
        temp_results = json.load(temp_reader)
    for key, value in temp_results.items():
        if key != 'ragchecker':  
            model_results[key] = value
    os.remove(temp_output_path)

with open(output_path, 'w', encoding='utf-8') as output_writer:
    json.dump(final_results, output_writer, indent=2, ensure_ascii=False)

print("All results saved to", output_path)
print("Final structure:")
print(json.dumps({k: list(v.keys()) for k, v in final_results.items()}, indent=2))