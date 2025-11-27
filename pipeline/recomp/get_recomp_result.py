from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from .run_extractive_compressor import get_contriever_scores
import pandas as pd
from ..common import recomp_contriever_path


def get_llm_input(model_path, input_data_df, device, sentence_top_k=20):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to('cuda:{}'.format(device))

    # get contriever scores
    contriever_scores = []

    for _, data in tqdm(input_data_df.iterrows(), total=len(input_data_df)):
        scores = get_contriever_scores(model, tokenizer, data, 'cuda:{}'.format(device), top_k=sentence_top_k)
        contriever_scores.append((scores, data['retrieved_docs']))

    return contriever_scores

def recomp_extractive_docs(query, docs, sentence_top_k=20):
    embedding_model_path = recomp_contriever_path
    final_df = {'query': {}, 'retrieved_docs': {}}
    final_df['query']['0'] = query
    # split to sentences
    final_df['retrieved_docs']['0'] = [{'text': sentence} for doc in docs for sentence in doc.page_content.split('. ')]
    # recomp format
    input_data_df = pd.DataFrame({"query": pd.Series(final_df["query"]),
                                "retrieved_docs": pd.Series(final_df["retrieved_docs"])})
    # comput contriever scores
    contriever_scores = get_llm_input(embedding_model_path, input_data_df, device='0', sentence_top_k=sentence_top_k)
    scores, recomp_data = contriever_scores[0]
    sorted_recomp_data = [x for _, x in sorted(zip(scores, recomp_data), key=lambda pair: pair[0], reverse=True)]
    return [sentence['text'] for sentence in sorted_recomp_data[:sentence_top_k]]