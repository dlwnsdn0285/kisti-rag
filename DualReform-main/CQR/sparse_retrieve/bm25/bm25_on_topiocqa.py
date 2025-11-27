from re import T
from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import argparse
import os
from utils import check_dir_exist_or_build
from os import path
from os.path import join as oj
import toml
import numpy as np
import json
from pyserini.search.lucene import LuceneSearcher
import pytrec_eval

def main():
    args = get_args()
    print(args)
    
    query_list = []
    qid_list = []
    with open(args.input_query_path, "r") as f:
        data = f.readlines()
        data_ids = [json.loads(data[i])['id'] for i in range(len(data))]
        n = len(set(data_ids))

    # if args.input_query_path_2 is not None:
    #     with open(args.input_query_path_2, "r") as f2:
    #         data_2 = f2.readlines()

    if args.input_query_path_2:
        with open(args.input_query_path_2, "r") as f2:
            data_2 = f2.readlines()
        n = min(n,len(data_2))
        data_2_dict = [json.loads(data_2[i]) for i in range(len(data_2))]
        data_2_dict = {l['id']:l for l in data_2_dict}

    if args.input_query_path_3:
        with open(args.input_query_path_3, "r") as f3:
            data_3 = f3.readlines()
        n = min(n,len(data_3))
        data_3_dict = [json.loads(data_3[i]) for i in range(len(data_3))]
        data_3_dict = {l['id']:l for l in data_3_dict}
        
    if args.input_query_path_4:
        with open(args.input_query_path_4, "r") as f4:
            data_4 = f4.readlines()
        n = min(n,len(data_4))
        data_4_dict = [json.loads(data_4[i]) for i in range(len(data_4))]
        data_4_dict = {l['id']:l for l in data_4_dict}

    if args.input_query_path_5:
        with open(args.input_query_path_5, "r") as f5:
            data_5 = f5.readlines()
        n = min(n,len(data_5))
        data_5_dict = [json.loads(data_5[i]) for i in range(len(data_5))]
        data_5_dict = {l['id']:l for l in data_5_dict}

    if args.input_query_path_6:
        with open(args.input_query_path_6, "r") as f6:
            data_6 = f6.readlines()
        n = min(n,len(data_6))
        data_6_dict = [json.loads(data_6[i]) for i in range(len(data_6))]
        data_6_dict = {l['id']:l for l in data_6_dict}
    


    if args.use_PRF:
        with open(args.PRF_file, 'r') as f:
            PRF = f.readlines()
        assert(len(data) == len(PRF))
        
    for i in range(n):
        data[i] = json.loads(data[i])
        if args.query_type == "raw":
            query = data[i]["query"]
        elif args.query_type == "rewrite":
            query = data[i]['rewrite'] #+ ' ' + data[i]['answer']

        # ! back-retrieval
        elif "this_answer_only" in args.query_type:  # "this_answer_only" "this_answer" "this_answer_truth_rewrite"
            query = data[i]['answer'] #+ ' ' + data[i]['answer']
            
        # ! back-retrieval
        elif args.query_type == "this_answer_truth_rewrite": # T5-trained rewrite
            query = [data[i]["original_oracle_utt_text"], data[i]["answer"]]
            query = " ".join(query)

        # ! back-retrieval
        elif args.query_type == "this_answer": # 
            query = [data[i]["cur_utt_text"], data[i]["answer"]]
            query = " ".join(query)
            
            
        # ! for back-retrieval: originalCTX_resp_bwd#_fwd#
        elif "originalCTX+resp" in args.query_type: # args.query_type.startswith("originalCTX+resp"): 
            bwd_n, fwd_n = args.query_type.split("_")[1], args.query_type.split("_")[2]
            bwd_n, fwd_n = int(bwd_n.split("bwd")[-1]), int(fwd_n.split("fwd")[-1])

            # get bwd_ctx
            curr_turn_dt = data[i]
            bwd_ctx = [item for pair in zip(curr_turn_dt['history_query'], curr_turn_dt['history_answer'])
                         for item in pair] + [curr_turn_dt['cur_utt_text']]
            bwd_ctx = bwd_ctx[-bwd_n:] if bwd_n>0 else []

            # get fwd_ctx
            i_add = (fwd_n+1)//2
            while i_add > 0:
                if i+i_add > n-1: # end of data?
                    i_add -= 1
                    fwd_n = 2*i_add # when i_add is adjusted, see ctx until last
                    continue
                
                fwd_turn_dt = json.loads(data[i+i_add]) # data[i+i_add]
                if fwd_turn_dt['conv_id'] != data[i]['conv_id']:
                    i_add -= 1
                    fwd_n = 2*i_add # when i_add is adjusted, see ctx until last
                    # print(i_add, fwd_n)
                else: # found same conv_id!
                    break 

            fwd_turn_dt = json.loads(data[i+i_add]) if i_add>0 else data[i] # data[i+i_add]
            # fwd_ctx = [item for pair in zip(fwd_turn_dt['history_query'], fwd_turn_dt['history_answer'])
            #              for item in pair] + [fwd_turn_dt['original_oracle_utt_text']] + [fwd_turn_dt['answer']]
            fwd_ctx = [item for pair in zip(fwd_turn_dt['history_query'], fwd_turn_dt['history_answer'])
                         for item in pair] + [fwd_turn_dt['cur_utt_text']] + [fwd_turn_dt['answer']]
            # print("fwd_ctx: ", len(fwd_ctx))
            # print(-i_add*2, -i_add*2+fwd_n, fwd_n)
            # print(len(fwd_ctx[-2 : ]))
            if -i_add*2+fwd_n == 0: # 
                fwd_ctx = fwd_ctx[-i_add*2 :] if fwd_n>0 else []
            else:
                fwd_ctx = fwd_ctx[-i_add*2 : -i_add*2+fwd_n] if fwd_n>0 else []
            # fwd_ctx = fwd_ctx[:-1] if fwd_n%2 == 1 else fwd_ctx # if odd, ctx: until Q; if even, ctx: until QA
            # print("fwd_ctx: ", len(fwd_ctx))

            query = bwd_ctx + [data[i]['answer']] + fwd_ctx
            # print(len(query), query)

            query = ' '.join(query)

        # ! for back-retrieval: response_bwd#_fwd#
        elif "resp" in args.query_type: # args.query_type.startswith("resp"): 
            bwd_n, fwd_n = args.query_type.split("_")[1], args.query_type.split("_")[2]
            bwd_n, fwd_n = int(bwd_n.split("bwd")[-1]), int(fwd_n.split("fwd")[-1])

            # get bwd_ctx
            curr_turn_dt = data[i]
            bwd_ctx = [item for pair in zip(curr_turn_dt['history_rewrite'], curr_turn_dt['history_answer'])
                         for item in pair] + [curr_turn_dt['original_oracle_utt_text']]
            bwd_ctx = bwd_ctx[-bwd_n:] if bwd_n>0 else []

            # get fwd_ctx
            i_add = (fwd_n+1)//2
            while i_add > 0:
                if i+i_add > n-1: # end of data?
                    i_add -= 1
                    fwd_n = 2*i_add # when i_add is adjusted, see ctx until last
                    continue
                
                fwd_turn_dt = json.loads(data[i+i_add]) # data[i+i_add]
                if fwd_turn_dt['conv_id'] != data[i]['conv_id']:
                    i_add -= 1
                    fwd_n = 2*i_add # when i_add is adjusted, see ctx until last
                    # print(i_add, fwd_n)
                else: # found same conv_id!
                    break 

            fwd_turn_dt = json.loads(data[i+i_add]) if i_add>0 else data[i] # data[i+i_add]
            fwd_ctx = [item for pair in zip(fwd_turn_dt['history_rewrite'], fwd_turn_dt['history_answer'])
                         for item in pair] + [fwd_turn_dt['original_oracle_utt_text']] + [fwd_turn_dt['answer']]
            # print("fwd_ctx: ", len(fwd_ctx))
            # print(-i_add*2, -i_add*2+fwd_n, fwd_n)
            # print(len(fwd_ctx[-2 : ]))
            if -i_add*2+fwd_n == 0: # 
                fwd_ctx = fwd_ctx[-i_add*2 :] if fwd_n>0 else []
            else:
                fwd_ctx = fwd_ctx[-i_add*2 : -i_add*2+fwd_n] if fwd_n>0 else []
            # fwd_ctx = fwd_ctx[:-1] if fwd_n%2 == 1 else fwd_ctx # if odd, ctx: until Q; if even, ctx: until QA
            # print("fwd_ctx: ", len(fwd_ctx))

            query = bwd_ctx + [data[i]['answer']] + fwd_ctx
            # print(len(query), query)

            query = ' '.join(query)

        elif args.query_type == "decode":
            query = data[i]['oracle_utt_text']
            if args.eval_type == "answer":
                data_2[i] = json.loads(data_2[i])
                query = data_2[i]['answer_utt_text']
            elif args.eval_type == "oracle+answer":
                data_2[i] = json.loads(data_2[i])
                query = query + ' ' + data_2[i]['answer_utt_text']

        if "selfask" in args.query_type:
            selfask_info = data[i]["oracle_utt_text"]
            sample_id = data[i]['id']
            if args.input_query_path_2:
                # data_2[i] = json.loads(data_2[i])
                # selfask_info += " " + data_2[i]["oracle_utt_text"]
                selfask_info += " " + data_2_dict[sample_id]["oracle_utt_text"]
            if args.input_query_path_3:
                # data_3[i] = json.loads(data_3[i])
                # selfask_info += " " + data_3[i]["oracle_utt_text"]
                selfask_info += " " + data_3_dict[sample_id]["oracle_utt_text"]
            if args.input_query_path_4:
                # data_4[i] = json.loads(data_4[i])
                # selfask_info += " " + data_4[i]["oracle_utt_text"]
                selfask_info += " " + data_4_dict[sample_id]["oracle_utt_text"]
            if args.input_query_path_5:
                # data_5[i] = json.loads(data_5[i])
                # selfask_info += " " + data_5[i]["oracle_utt_text"]
                selfask_info += " " + data_5_dict[sample_id]["oracle_utt_text"]
            if args.input_query_path_6:
                # data_6[i] = json.loads(data_6[i])
                # selfask_info += " " + data_6[i]["oracle_utt_text"]
                selfask_info += " " + data_6_dict[sample_id]["oracle_utt_text"]
            
            if args.selfask_1part:
                selfask_info = selfask_info[:len(selfask_info)//2]
            elif args.selfask_2part: 
                selfask_info = selfask_info[len(selfask_info)//2:]

            query = query + ': ' + selfask_info

        print(len(query), query)
        query_list.append(query)
        if "sample_id" in data[i]:
            qid_list.append(data[i]['sample_id'])
        else:
            qid_list.append(data[i]['id'])
        
    # print("query_list: ", query_list)
    # pyserini search
    print(f"{len(qid_list)} number of queries")

    searcher = LuceneSearcher(args.index_dir_path)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    hits = searcher.batch_search(query_list, qid_list, k = args.top_k, threads = 20)
    # qid_list is for key search
    
    output_f = args.output_f if args.output_f is not None else "bm25_t5_{}_res.trec".format(args.eval_type)
    with open(oj(args.output_dir_path, output_f), "w") as f:
        for qid in qid_list:
            for i, item in enumerate(hits[qid]):
                f.write("{} {} {} {} {} {} {}".format(qid,
                                                "Q0",
                                                item.docid[3:],
                                                i+1,
                                                -i - 1 + 200,
                                                item.score,
                                                "bm25"
                                                ))
                f.write('\n')


    res = print_res(oj(args.output_dir_path, output_f), 
                    args.gold_qrel_file_path, args.rel_threshold)
    return res


def print_res(run_file, qrel_file, rel_threshold):
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    # print(run_data)
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    
    for line in qrel_data:
        line = line.strip().split()
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
    
    for line in run_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(line[4])
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

    res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "NDCG@3": np.average(ndcg_3_list),
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@20": np.average(recall_20_list),
            "Recall@100": np.average(recall_100_list), 
        }

    
    logger.info("---------------------Evaluation results:---------------------")    
    logger.info(res)
    return res



def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_query_path", type=str, default="output/topiocqa/QR/dev_QRIR_oracle_prefix.json")
    # parser.add_argument("--input_query_path_2", type=str, default="output/topiocqa/QR/dev_QRIR_answer_prefix.json")
    # parser.add_argument("--index_dir_path", type=str, default="datasets/topiocqa/indexes/bm25")
    # parser.add_argument("--output_dir_path", type=str, default="output/topiocqa/bm25")
    # parser.add_argument("--gold_qrel_file_path", type=str, default="datasets/topiocqa/dev_gold.trec")
    
    
    parser.add_argument("--input_query_path", type=str, default="/data2/../nlp_data/infocqr_data/topiocqa/test_chatgpt_icl_WOpssg_original_seed0_temp8_p8_sampled_lamma_bs64_lr5e-4_Full1000.jsonl")
    parser.add_argument("--input_query_path_2", type=str, default=None)
    parser.add_argument("--input_query_path_3", type=str, default=None)
    parser.add_argument("--input_query_path_4", type=str, default=None)
    parser.add_argument("--input_query_path_5", type=str, default=None)
    parser.add_argument("--input_query_path_6", type=str, default=None)    
    parser.add_argument("--index_dir_path", type=str, default="/data2/../nlp_data/topiocqa/indexes/bm25")
    parser.add_argument("--output_dir_path", type=str, default="/data2/../nlp_data/convgqr/bm25/qrecc_trained")
    parser.add_argument("--output_f", type=str, default=None)
    parser.add_argument("--gold_qrel_file_path", type=str, default="/data/../nlp_data/topiocqa/dev_gold.trec")
    parser.add_argument("--use_PRF", type=bool, default=False)

    parser.add_argument("--selfask_1part", type=bool, default=False)
    parser.add_argument("--selfask_2part", type=bool, default=False)
    parser.add_argument("--query_type", type=str, default="decode")
    parser.add_argument("--eval_type", type=str, default="oracle+answer")
    parser.add_argument("--bm25_k1", type=float, default="0.9")
    parser.add_argument("--bm25_b", type=float, default="0.4")
    parser.add_argument("--top_k", type=int, default="100")
    parser.add_argument("--rel_threshold", type=int, default="1")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()
