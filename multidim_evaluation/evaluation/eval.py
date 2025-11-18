import os
import warnings

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore', message='.*generation flags.*')

import re
import os
import json
import string
import warnings
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from evaluate import load

# Suppress the specific tokenization warning
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*", category=FutureWarning)
# load_llama_model('Llama3.1_8b')

# Ensure the output directory exists
from .lib import (
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
    get_config_file_path_from_name_or_path,
)
from .metrics.drop_answer_em_f1 import DropAnswerEmAndF1
from .metrics.support_em_f1 import SupportEmF1Metric
from .metrics.answer_support_recall import AnswerSupportRecallMetric
from .metrics.squad_answer_em_f1 import SquadAnswerEmF1Metric

# Load BERTScore with explicit tokenizer configuration
bertscore = load("bertscore", model_type="bert-base-multilingual-cased")

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def answer_extractor(potentially_cot: str) -> str:
    if potentially_cot.startswith('"') and potentially_cot.endswith('"'):
        potentially_cot = potentially_cot[1:-1]

    cot_regex = re.compile(".* answer is:? (.*)\\.?")
    match = cot_regex.match(potentially_cot)
    if match:
        output = match.group(1)
        if output.endswith("."):
            output = output[:-1]
    else:
        output = potentially_cot

    return output

def calculate_acc(prediction, ground_truth):
    for gt in ground_truth:
        if gt in prediction:
            return 1
    return 0

def calculate_bleu(reference, hypothesis):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference], hypothesis, smoothing_function=smoothie)

def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

def evaluate_by_dicts(input_path, output_path):
    metrics = [SquadAnswerEmF1Metric()]
     
    total_acc = 0
    total_lines = 0
    total_rouge1 = 0
    total_rougel = 0
    total_bert = 0

    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        data = data['results']
        for data_dic in data:
            query, ground_truth, prediction = data_dic['query'], data_dic['gt_answer'], data_dic['response']
            
            # Suppress warnings for BERTScore computation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                bert_score = bertscore.compute(references=[ground_truth], predictions=[prediction], lang='ko')

            ground_truth, prediction = [ground_truth], [prediction]          

            assert isinstance(prediction, (str, list))
            if isinstance(prediction, str):
                if prediction.strip().startswith("[") or prediction.strip().endswith("]"):
                    prediction = [e for e in prediction.replace('"', "").replace("[", "").replace("]", "").split(",")]
                else:
                    prediction = [prediction]

            assert isinstance(prediction, (list, tuple))
            prediction = [str(e) for e in prediction]
            prediction = [answer_extractor(_prediction) for _prediction in prediction]

            normalized_prediction = normalize_answer(prediction[0])
            normalized_ground_truth = [normalize_answer(i) for i in ground_truth[0]]

            acc = calculate_acc(normalized_prediction, normalized_ground_truth)
            total_acc += acc

            rouge_scores = calculate_rouge(" ".join(normalized_ground_truth), normalized_prediction)
            total_rouge1 += rouge_scores['rouge1'].fmeasure
            total_rougel += rouge_scores['rougeL'].fmeasure

            total_bert += bert_score['f1'][0]

            total_lines += 1
            try : 
                metrics[0](prediction, ground_truth)
            except : 
                pass
            
        total_acc = total_acc / total_lines
        total_rouge1 = total_rouge1 / total_lines
        total_rougel = total_rougel / total_lines
        total_bert = total_bert / total_lines
        
        evaluation_results = metrics[0].get_metric()
        evaluation_results['acc'] = total_acc
        evaluation_results['rouge1'] = total_rouge1
        evaluation_results['rougel'] = total_rougel
        evaluation_results['bert_f1'] = total_bert

    save_results(evaluation_results, output_path)

def save_results(results_dict, output_path):
    with open(output_path, "w") as file:
        json.dump(results_dict, file, indent=4)

def recalculate_metrics(input_path, output_path):
    metrics = [SquadAnswerEmF1Metric()]
     
    total_acc = 0
    total_lines = 0
    total_rouge1 = 0
    total_rougel = 0
    total_bert = 0
    total_lcs = 0
    total_ngram_overlap = 0
    total_jaccard = 0
    
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        data_dic = data['results']
        for line in data_dic:
            data = line['results']
            query, ground_truth, prediction = data['query'], data['gt_answer'], data['response']
            
            # Suppress warnings for BERTScore computation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                bert_score = bertscore.compute(references=ground_truth, predictions=prediction, lang='ko')

            prediction = [prediction]            

            assert isinstance(prediction, (str, list))
            if isinstance(prediction, str):
                if prediction.strip().startswith("[") or prediction.strip().endswith("]"):
                    prediction = [e for e in prediction.replace('"', "").replace("[", "").replace("]", "").split(",")]
                else:
                    prediction = [prediction]

            assert isinstance(prediction, (list, tuple))
            prediction = [str(e) for e in prediction]
            prediction = [answer_extractor(_prediction) for _prediction in prediction]

            normalized_prediction = normalize_answer(prediction[0])
            normalized_ground_truth = [normalize_answer(i) for i in ground_truth]

            acc = calculate_acc(normalized_prediction, normalized_ground_truth)
            total_acc += acc

            rouge_scores = calculate_rouge(" ".join(normalized_ground_truth), normalized_prediction)
            total_rouge1 += rouge_scores['rouge1'].fmeasure
            total_rougel += rouge_scores['rougeL'].fmeasure

            total_bert += bert_score['f1'][0]

            total_lines += 1
            try:
                metrics[0](prediction, ground_truth)
            except:
                pass
            
    total_acc = total_acc / total_lines
    total_lcs = total_lcs / total_lines
    total_ngram = total_ngram_overlap / total_lines
    total_jaccard = total_jaccard / total_lines
    total_rouge1 = total_rouge1 / total_lines
    total_rougel = total_rougel / total_lines
    total_bert = total_bert / total_lines
    
    evaluation_results = metrics[0].get_metric()
    evaluation_results['acc'] = total_acc
    evaluation_results['lcs'] = total_lcs
    evaluation_results['ngram'] = total_ngram
    evaluation_results['jaccard'] = total_jaccard
    evaluation_results['rouge1'] = total_rouge1
    evaluation_results['rougel'] = total_rougel
    evaluation_results['bert_f1'] = total_bert

    save_results(evaluation_results, output_path)

# Example usage:
if __name__ == '__main__':
    input_path = './data/ragcheck_with_response.json'
    output_path = './data/results/response_eval.json'
    evaluate_by_dicts(input_path, output_path)