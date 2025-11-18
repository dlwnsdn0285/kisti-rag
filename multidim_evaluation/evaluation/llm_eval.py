import os
import warnings

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore', message='.*generation flags.*')

import json
import re
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
from .qwen_model import *

SYS_PROMPT = "You are a fair evaluator language model."

def extract_score_from_text(text):
    """
    Extracts the score from various patterns in the given text.
    Parameters:
    text (str): The input text containing a score.
    Returns:
    int: The extracted score as an integer (0 or 1), or 0 if no valid score is found.
    """
    # Try different patterns to extract the score
    patterns = [
        r'\[RESULT\]\s*(\d+)',  # [RESULT] 1
        r'\{(\d+)\}',           # {1}
        r'output\s+(\d+)',      # output 1
        r'따라서.*?(\d+)',       # 따라서... 1
        r'output은\s*(\d+)',     # output은 1
        r'결과.*?(\d+)'          # 결과... 1
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            score = int(match.group(1))
            # Only return 0 or 1, anything else becomes 0
            return score if score in [0, 1] else 0
    
    # If no pattern matches, try to find any single digit 0 or 1
    single_digit = re.findall(r'\b[01]\b', text)
    if single_digit:
        return int(single_digit[-1])  # Return the last found 0 or 1
    
    # Default to 0 if nothing found
    return 0

def llm_eval_generate(formatted_prompt):
    messages = [{"role":"system","content":SYS_PROMPT},{"role":"user","content":formatted_prompt}]
    return generate_response(messages, max_new_tokens=500, do_sample=False, temperature=None, top_p=None)

def format_query_prompt(query, gt, pd):
    PROMPT = f"""
### Task Description:
1. Determine whether the predicted answer provides the similar level of detail and explanation as the reference answer (ground truth answer).
2. If the reference answer includes specific steps, procedures, or details, the predicted answer must also include similarly specific information.
3. General or vague responses that do not cover the key details will be considered incorrect, output "0".
4. If the predicted answer only briefly mentions the topic without providing the necessary level of detail or fails to explain the key points, output "0".
5. If the predicted answer is concise but covers all critical steps or points provided in the reference answer, output "1".
6. Never generate any other content besides the binary score.
7. The output format should be strictly: {{either 0 or 1}}.

### Question :
{query}

### Predicted:
{pd}

### Reference:
{gt}

### Evaluation Criteria:
1. If the predicted answer provides a similar level of detail, covers the key points, and aligns with the context of the query and the reference answer, output 1.
2. If the predicted answer is concise but accurately captures the core concept or key term(s) from the reference answer in response to the query, output 1.
3. If the predicted answer is too general, omits important details, lacks critical steps, output 0.
4. If the predicted answer does not share key terms or concepts with the reference answer or deviates significantly from the context, output 0.
5. If the predicted answer is empty(e.g "{{}}"), output 0.
"""
    return PROMPT

def RAG_eval_w_LLM(eval_path, verbose=False, llm_model="Qwen/Qwen3-30B-A3B-Instruct-2507"):
    # Load the model first
    print(f"Loading model: {llm_model}")
    load_qwen_model(llm_model)
    print("Model loaded successfully!")
    
    total_score = 0
    error = 0
    with open(eval_path, 'r', encoding='utf-8') as eval_file:
        data = json.load(eval_file)
    data_dic = data['results']
    
    # tqdm 추가 - 전체 데이터에 대한 진행바
    for idx, data in enumerate(tqdm(data_dic, desc="Evaluating", unit="sample")):
        # Initialize variables before try block
        generated_eval = "Error occurred during evaluation"
        score = 0
        
        try:
            question = data['query']
            groundtruth = data['gt_answer']
            prediction = data['response']
            print(f'Question:{question}, GTAnswer:{groundtruth}, Prediction:{prediction}')
            prompt = format_query_prompt(question, groundtruth, prediction)
            generated_eval = llm_eval_generate(prompt)
            print(f'Response: {generated_eval}')
            score = extract_score_from_text(generated_eval)
        except Exception as e:
            error += 1
            print(f'\nError # {error}: {str(e)}')
            score = 0
        
        total_score += int(score)
            
        result_dict = {
            "index": idx + 1,
            "question": question,
            "ground_truth": groundtruth,
            "generated_answer": prediction,
            "score": score,
            "current_avg_score": round(total_score / (idx + 1), 4)
        }
        
        if verbose:
            print('=' * 50)
            print(f'# {idx + 1} Processing')
            print(f'Question : {question}')
            print(f'Ground Truth : {groundtruth}')
            print(f'Generated Answer: {prediction}')
            print(f'Generated Eval: {generated_eval}')
            print(f'Score : {score}')
            print(f'Current Avg Score: {total_score / (idx + 1):.4f}')
            print('=' * 50 + '\n')
    
    final_avg_score = total_score / (idx + 1)
    print(f'Final Avg Score: {final_avg_score:.2f}')
    return final_avg_score

if __name__ == '__main__':
    input_path = './data/ragcheck_with_response.json'
    # You can specify the model you want to use
    RAG_eval_w_LLM(input_path, verbose=True, llm_model='Qwen/Qwen3-Next-80B-A3B-Instruct')