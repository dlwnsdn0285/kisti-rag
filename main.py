from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever

from datetime import datetime

from pipeline.chunking.simple import get_simple_retriever
from pipeline.chunking.sentence_parent import get_sentence_parent_retriever, delete_sentence_parent_retriever_data
from pipeline.infer.infer import eval_retriever, eval_full_chain
from pipeline.util.embed import delete_embeddings, remove_small_chunks
from pipeline.common import setup_logger
from pipeline.eval.eval import evaluate_by_dicts, recalculate_metrics
from pipeline.util.kisti_data import sample_kisti, get_sample_paper, get_sample_qa
from pipeline.util.dense_runnable import DenseRetrieverWithHyde
from pipeline.common import input_path, output_path, input_path_ragchecker

import argparse

DENSE='dense'
SPARSE='sparse'
ENSEMBLE='ensemble'

def create_sample_data(sample_paper_num, sample_qa_num):
    sample_kisti(sample_paper_num, sample_qa_num)
    print(len(get_sample_paper()))
    print(len(get_sample_qa()))

def retrieval_chain(retriever_type, args):
    base_subdirectory = f'eval_retriever/{retriever_type}'
    eval_logger = setup_logger('eval_retriever', subdirectory=base_subdirectory)
    hyde_logger = setup_logger('hyde', subdirectory=base_subdirectory) if args.hyde==True else None

    ks = [4, 8]

    sparse_retriever = get_simple_retriever('bm25', 500, 50)
    dense_retriever = DenseRetrieverWithHyde(get_simple_retriever('dense', 500, 50), hyde=args.hyde, hyde_logger=hyde_logger)

    retriever_map = {
        'sparse': sparse_retriever,
        'dense': dense_retriever,
        'ensemble': EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[0.5, 0.5]
        )
    }

    if retriever_type not in retriever_map:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    for k in ks:
        result = eval_retriever(
            retriever_map[retriever_type], 
            k, 
            hyde=args.hyde, 
            eval_logger=eval_logger, 
            hyde_logger=hyde_logger
        )
        eval_logger.info(f"Result: retriever={retriever_type}, k={k}, hyde={args.hyde}, result={result}")
        print(f'{retriever_type}', k, result)

def full_chain(args):

    # retriever 분기 (sparse/dense/ensemble/qudar) -> 정해진 하나의 retriever : decided_retriever
    ks = [8]

    sparse_retriever = get_simple_retriever('bm25', 500, 50)
    dense_retriever = DenseRetrieverWithHyde(get_simple_retriever('dense', 500, 50), hyde=args.hyde, hyde_logger=hyde_logger)

    decided_retriever = None #####

    base_subdirectory = f'eval_full_chain/{decided_retriever}'
    eval_logger = setup_logger('eval_full_chain', subdirectory=base_subdirectory)
    hyde_logger = setup_logger('hyde', subdirectory=base_subdirectory) if args.hyde==True else None


    for k in ks: # [MODIFIED] args, decided_retriever added
        result=eval_full_chain(args, decided_retriever, k, input_path=input_path, output_path=output_path, input_path_ragchecker=input_path_ragchecker, eval_logger=eval_logger, hyde_logger=hyde_logger)
        print(k, result)

if __name__ == '__main__':
    #create_sample_data(3,30)
    #retrieval_chain(ENSEMBLE, args)
    
    QUDAR_CHOICES = ["QUDAR_simple_rrf", "QUDAR_simple_equal", "QUDAR_confidence", "QUDAR_llm"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyde', type=bool, help='Enable HyDE', default=False)
    parser.add_argument('--rerank', type=bool, help='Enable Reranking', default=False)
    parser.add_argument('--adaptive', type=bool, help='Enable adaptive retrieval', default=False)
    parser.add_argument('--retriever', type=str, choices=[DENSE, SPARSE, ENSEMBLE] + QUDAR_CHOICES, default=DENSE)
    #parser.add_argument('--generation', type=str, choices =['base', 'recomp', 'ext2gen'], default='base')
    parser.add_argument('--recomp', type=bool, help='Enable recomp', default=False)
    parser.add_argument('--ext2gen', type=bool, help='Enable ext2gen', default=False)
    args = parser.parse_args()
    
    full_chain(args)
