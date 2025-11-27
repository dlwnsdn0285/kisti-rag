from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from pipeline.chunking.simple import get_simple_retriever
from pipeline.common import embedding_dir, embedding_function
from ..llm import format_hyde_prompt, hyde_generate

# OS/OD/ES/ED 키 스키마
SRC_KEYS = [
    ("sparse_docs",     "sparse_scores",     "sparse_scores_normalized"),     # OS
    ("dense_docs",      "dense_scores",      "dense_scores_normalized"),      # OD
    ("pgt_sparse_docs", "pgt_sparse_scores", "pgt_sparse_scores_normalized"), # ES
    ("pgt_dense_docs",  "pgt_dense_scores",  "pgt_dense_scores_normalized"),  # ED
]

KEY_ALIAS = {
    "sparse": "original_sparse",
    "dense": "original_dense",
    "pgt_sparse": "expanded_sparse",
    "pgt_dense": "expanded_dense",
}


# --- 공통 유틸 ---
def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0: return x
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x, dtype=float)

def _get_norm_scores(qentry: Dict[str, Any], raw_key: str, norm_key: str) -> np.ndarray:
    if norm_key in qentry and qentry[norm_key] is not None:
        return np.asarray(qentry[norm_key], dtype=float)
    raw = np.asarray(qentry.get(raw_key, []), dtype=float)
    return _minmax(raw)

def _accumulate(doc2score: Dict[str, float], docs: List[str], scores: np.ndarray, w: float) -> None:
    if not docs or scores is None or w == 0: return
    n = min(len(docs), len(scores))
    for i in range(n):
        d, s = docs[i], float(scores[i])
        doc2score[d] = doc2score.get(d, 0.0) + w * s

def _stable_softmax(x, tau=1.0, eps=1e-9):
    x = np.asarray(x, dtype=float)
    z = (x - np.max(x)) / max(tau, eps)
    e = np.exp(z); s = e.sum()
    return e / s if s > 0 else np.ones_like(x) / len(x)

def _top_margin(scores: np.ndarray) -> float:
    if scores.size == 0: return 0.0
    s = -np.sort(-np.asarray(scores, dtype=float))
    return float(s[0]) if s.size == 1 else max(float(s[0]) - float(s[1]), 0.0)

def _build_llm_input_from_qentry(qentry: Dict[str, Any]) -> Dict[str, str]:
    llm_input = {}

    for docs_key, _, _ in SRC_KEYS:
        docs = qentry.get(docs_key) or []
        if not docs:
            continue

        base = docs_key.replace("_docs", "")
        llm_key = KEY_ALIAS.get(base, base)
        llm_input[llm_key] = docs[0]

    return llm_input

def llm_scoring_prompt(query, llm_input, verbose=False):

    passages = list(llm_input.values())

    user_content = f"""
You are an impartial evaluator of retrieval effectiveness.
Score each anonymous passage with an integer from 0 to 5 (inclusive) using ONLY the criteria below.
Return four integers separated by a space in order — nothing else.

### ** Scoring Criteria :**
5 = Direct hit (clearly answers the question)
4 = Very close (high likelihood correct answer is nearby)
3 = Somewhat close (related, partial answer or near-miss)
2 = Loosely related but likely off in ranking neighborhood
1 = Barely related; unlikely nearby
0 = Unrelated / off-track
---

### ** Output Format :**
Return four integers separated by a space :
- ** First number :** first passage score .
- ** Second number :** second passage score .
- ** Third number :** third passage score .
- ** Fourth number :** fourth passage score .
- Example output : 3 4 5 1
** Do not output any other text .**
---

### ** Given Data :**
- ** Question :** "{ query }"

You will see 4 anonymous retrieval results (Top1s), in random order:
- First Passage :  "{passages[0]}"
- Second Passage : "{passages[1]}"
- Third Passage :  "{passages[2]}"
- Fourth Passage : "{passages[3]}"

"""
    if verbose:
        print(user_content)

    return user_content

# --- QuDAR : Query-Wise Dual-Perspective Dynamic Adaptive Retrieval ---
def qudar_rrf(qentry: Dict[str, Any], rrf_k: int = 60) -> Dict[str, float]:
    doc2score = {}
    for (docs_key, _, _) in SRC_KEYS:
        docs = qentry.get(docs_key, []) or []
        for rank, docid in enumerate(docs, start=1):
            doc2score[docid] = doc2score.get(docid, 0.0) + 1.0 / (rrf_k + rank)
    return doc2score

def qudar_equal(qentry: Dict[str, Any], weights=(0.25, 0.25, 0.25, 0.25)) -> Dict[str, float]:
    doc2score = {}
    for w, (docs_key, raw_key, norm_key) in zip(weights, SRC_KEYS):
        docs = qentry.get(docs_key, []) or []
        scores = _get_norm_scores(qentry, raw_key, norm_key)
        _accumulate(doc2score, docs, scores, float(w))
    return doc2score

def qudar_confidence(qentry: Dict[str, Any], topk=10, tau=1.0) -> Dict[str, float]:
    margins = []
    tmp_scores = []
    for _, raw_key, norm_key in SRC_KEYS:
        s = _get_norm_scores(qentry, raw_key, norm_key)
        tmp_scores.append(s)
        margins.append(_top_margin(s[:topk]))
    w = _stable_softmax(margins, tau=tau)
    # print("confidence weight :", w)
    doc2score = {}
    for ww, (docs_key, _raw_key, _norm_key), s in zip(w, SRC_KEYS, tmp_scores):
        docs = qentry.get(docs_key, []) or []
        _accumulate(doc2score, docs, s, float(ww))
    return doc2score


def qudar_llm(query : str, qentry: Dict[str, Any]) -> Dict[str, float]:
    llm_input = _build_llm_input_from_qentry(qentry)

    if not llm_input:
        llm_scores: Dict[str, float] | None = None
    else:
        try:
            formatted_prompt = llm_scoring_prompt(query, llm_input)
            generated_text = hyde_generate(formatted_prompt, chunk_size=50) 
            values = list(map(float, generated_text.strip().split()))
            llm_scores = dict(zip(("original_sparse", "original_dense", "expanded_sparse", "expanded_dense"), values))
        except Exception:
            llm_scores = None
    
    if not llm_scores:
        w = np.array([0.25, 0.25, 0.25, 0.25], float)
    else:
        s = np.array(
            [llm_scores.get(k, 0.0) for k in ("original_sparse", "original_dense", "expanded_sparse", "expanded_dense")],
            float,
        )
        s = np.where(s < 0, 0, s)  # 음수는 0으로 클리핑
        w = _stable_softmax(s, tau=3)

    doc2score: Dict[str, float] = {}
    for ww, (docs_key, raw_key, norm_key) in zip(w, SRC_KEYS):
        docs = qentry.get(docs_key, []) or []
        scores = _get_norm_scores(qentry, raw_key, norm_key)
        _accumulate(doc2score, docs, scores, float(ww))

    return doc2score


# --- 최종 TOP-k 선택 ---
def qudar_to_topk_list(query : str, qentry: Dict[str, Any], k: int, strategy: str,
                      *, rrf_k=60, equal_weights=(0.25,0.25,0.25,0.25),
                      conf_topk=10, conf_tau=2.0, llm_scores=None) -> List[str]:
    if strategy == "QUDAR_simple_rrf":
        # print('QUDAR_simple_rrf... ')
        merged = qudar_rrf(qentry, rrf_k=rrf_k)
    elif strategy == "QUDAR_simple_equal":
        # print('QUDAR_simple_equal... ')
        merged = qudar_equal(qentry, weights=equal_weights)
    elif strategy == "QUDAR_confidence":
        # print('QUDAR_confidence... ')
        merged = qudar_confidence(qentry, topk=conf_topk, tau=conf_tau)
    elif strategy == "QUDAR_llm":
        merged = qudar_llm(query, qentry)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # 점수 내림차순 정렬 → 유니크 문서 k개
    items = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    top_docs = [docid for docid, _ in items[:k]]
    return top_docs

# qudar용 retriever setting - 기존과 다르게 score 필요

def load_sparse_retriever(retriever_type, k):
    """Initialize retrievers based on configuration"""
    print(f"Initializing {retriever_type} retriever (k={k})...")
    sparse_retriever = get_simple_retriever('bm25', 500, 50)
    return sparse_retriever

def sparse_retrieve_score(query, k, sparse_retriever):
    retriever = sparse_retriever    
    if retriever.vectorizer is None:
        raise ValueError("BM25 vectorizer not initialized.")
    processed_query = retriever.preprocess_func(query)
    return_docs = retriever.vectorizer.get_scores(processed_query)
    
    return_docs = [(doc.page_content, score) for doc, score in zip(retriever.docs, return_docs)]
    return sorted(return_docs, key=lambda x: x[1], reverse=True)[0:k]

def dense_retrieve_score(query, find_num):
    client = chromadb.PersistentClient(embedding_dir)
    name = 'kisti-500-50'
    db = Chroma(client=client, collection_name =name, embedding_function=embedding_function)    
    if db._collection.count() == 0:
        print('Embeddings not found')
        return
    result = db.similarity_search_with_relevance_scores(query, k=find_num)
    return [(doc.page_content, score) for doc, score in result]

# 4개 retrieval list 

def _split_docs_scores(pairs: List[List]) -> Tuple[List[str], List[float]]:
    """[[text, score], ...] -> (docs, scores)"""
    if not pairs:
        return [], []
    docs = [str(p[0]) for p in pairs]
    scores = [float(p[1]) for p in pairs]
    return docs, scores

def make_qentry_from_lists(
    os_list: List[List] = None,                  # original sparse: [[text, score], ...]
    od_list: List[List] = None,                  # original dense
    es_list: List[List] = None,           # expanded sparse (없으면 None/빈 리스트)
    ed_list: List[List] = None            # expanded dense
) -> Dict[str, Any]:
    os_list = os_list or []
    od_list = od_list or []
    es_list = es_list or []
    ed_list = ed_list or []

    os_docs, os_scores = _split_docs_scores(os_list)
    od_docs, od_scores = _split_docs_scores(od_list)
    es_docs, es_scores = _split_docs_scores(es_list)
    ed_docs, ed_scores = _split_docs_scores(ed_list)

    qentry = {
        "sparse_docs":        os_docs, "sparse_scores":       os_scores,
        "dense_docs":         od_docs, "dense_scores":        od_scores,
        "pgt_sparse_docs":    es_docs, "pgt_sparse_scores":   es_scores,
        "pgt_dense_docs":     ed_docs, "pgt_dense_scores":    ed_scores,
        # *_scores_normalized 키는 없어도 qudar에서 per-query min-max로 보정
    }
    return qentry

def make_qudar_retriever(strategy, pool_k=10000):
    sparse_for_score = load_sparse_retriever('bm25', k=pool_k)

    def qudar_retriever(query, k):
        os_pairs = sparse_retrieve_score(query, pool_k, sparse_for_score)
        od_pairs = dense_retrieve_score(query, pool_k)

        # generating expanded query (with hyde)
        # it takes time.. preprocessing is maybe better 
        formatted_prompt = format_hyde_prompt(query, chunk_size=256)
        expanded_query = hyde_generate(formatted_prompt, chunk_size=256)
        
        es_pairs = sparse_retrieve_score(expanded_query, pool_k, sparse_for_score)
        ed_pairs = dense_retrieve_score(expanded_query, pool_k)
        

        qentry = make_qentry_from_lists(
            os_list=os_pairs, od_list=od_pairs,
            es_list=es_pairs, ed_list=ed_pairs
        )

        top_texts = qudar_to_topk_list(query, qentry, k=k, strategy=strategy)

        # → Document list로 변환
        return [
            Document(page_content=t, metadata={"doc_id": hash(t)})
            for t in top_texts
        ]

    return qudar_retriever

