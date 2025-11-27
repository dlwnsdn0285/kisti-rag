import os
import json
import argparse
import logging
from tqdm import tqdm, trange

from utils.indexing_utils import DenseIndexer, DocumentCollection
from utils import get_logger
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


def read_qrecc_data(dataset, read_by="all", is_test=False):
    examples = []
    for data in tqdm(dataset):
        guid = f"{data['Conversation_no']}_{data['Turn_no']}"
        context = data['NewContext']
        assert len(context) % 2 == 0

        target_question = data["Question"]
        
        if read_by == "all":
            x = context + [target_question]
            x = " ".join(x)
        elif read_by == "all_without_this":
            x = context
            x = " ".join(x)
        elif read_by == "Truth_rewrite":
            x = data["Truth_rewrite"]
        elif read_by == "GPT_rewrite":
            x = data["GPT_rewrite"]
        elif read_by == "Editor_rewrite":
            x = data["Editor_rewrite"]
        elif read_by == "original":
            x = data["Question"]
        elif read_by == "this_answer":
            x = [data["Question"], data["Truth_answer"]]
            x = " ".join(x)
        elif read_by == "this_answer_truth_rewrite":
            x = [data["Truth_rewrite"], data["Truth_answer"]]
            x = " ".join(x)
        elif read_by == "this_answer_only":
            x = data["Truth_answer"]
        else:
            raise Exception("Unsupported option!")

        examples.append([guid, x])
        
        if is_test:
            logger.info(f"{guid}: {x}")
            if len(examples) == 10:
                break
        
    return examples


def read_qrecc_data_model_pred(dataset):
    examples = []
    for did in tqdm(dataset.keys()):
        new_did = "_".join(did.split("_")[-2:])
        examples.append([new_did, dataset[did]['pred']])
  
    return examples


def merge_scores(scores_list, topk):
    results = {}
    for rr in scores_list:
        for k, v in rr.items():
            if k not in results:
                results[k] = list(v.items())
            else:
                results[k].extend(list(v.items()))
                
    new_results = {}
    for k, v in results.items():
        new_results[k] = {}
        vv = sorted(v, key=lambda x: -x[1])
        for i in range(topk):
            pid, ss = vv[i]
            new_results[k][pid] = ss
            
    return new_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="qrecc")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--read_by', type=str, default="Truth_rewrite")
    parser.add_argument('--raw_data_path', type=str, default="/home/../CQR/datasets")
    parser.add_argument('--preprocessed_data_path', type=str, default="/data2/../nlp_data/convgqr/ance/test")
    parser.add_argument("--output_f", type=str, default=None)
    parser.add_argument('--dense_index_path', type=str, default="/home/../CQR/datasets/preprocessed/qrecc/dense_index")
    parser.add_argument('--data_file', type=str, default=None) # not in use
    parser.add_argument('--model_name_or_path', type=str, default='sentence-transformers/gtr-t5-base')

    # File paths
    parser.add_argument('--test_file_path', type=str, default=None, help='Path to the test file')
    parser.add_argument('--test_file_path_2', type=str, default=None, help='Path to the second test file')
    # convgqr setup
    parser.add_argument('--test_type', type=str, default="rewrite", help='Type of test') 
    parser.add_argument('--eval_type', type=str, default="oracle+answer", help='Evaluation type') # "oracle+answer" # "oracle" "oracle+answer" "answer"

    parser.add_argument('--prompt_type', type=str, default='zsl') # e.g., 'icl', 'zsl', etc.
    parser.add_argument('--instruct_pssg', type=str, default="original") # original, filter_irrelevant, summary, filter_irrelevant_summary
    parser.add_argument('--num_splits', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=100)
    args = parser.parse_args()
    
    if not os.path.exists(args.preprocessed_data_path):
        os.makedirs(args.preprocessed_data_path, exist_ok=True)
        
    fileHandler = logging.FileHandler(f"{args.preprocessed_data_path}/log.out", "a")
    formatter = logging.Formatter('%(asctime)s > %(message)s')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.info("logging start!")
    
    out_sfx = args.data_file.lstrip(args.split+"_").strip(".json") if args.data_file is not None else None 
    # read data
    # if args.instruct_pssg == 'n_return':
    #     dir_path = args.data_file.split('_v')[0] + '_' + args.prompt_type +\
    #              args.data_file.split(args.prompt_type)[-1].split('_post.json')[0]
    #     data = json.load(open(f"{args.raw_data_path}/{args.task}/{dir_path}/{args.data_file}", "r", encoding="utf-8"))
    # else:
    #     data = json.load(open(f"{args.raw_data_path}/{args.task}/{args.data_file}", "r", encoding="utf-8"))
    # data = json.load(open(f"{args.raw_data_path}/{args.task}/{args.data_file}", "r", encoding="utf-8"))

    # if args.read_by == "model":
    #     raw_examples = read_qrecc_data_model_pred(data)
    # else:
    #     raw_examples = read_qrecc_data(data, args.read_by)

    ########## ! imported codes ##########
    with open(args.test_file_path, 'r') as f:
        data = f.readlines()
        logger.info("Loading {} data file...".format(args.test_file_path))

    if args.test_file_path_2:
        with open(args.test_file_path_2, 'r') as f2:
            data_2 = f2.readlines()
        logger.info("Loading {} data file...".format(args.test_file_path_2))
    
    n = len(data)
    # n = int(args.use_data_percent * n)
    
    queries = []
    qids = []
    for i in trange(n):
        data[i] = json.loads(data[i])
        if 'id' in data[i]:
            sample_id = data[i]['id']
        else:
            sample_id = data[i]['sample_id']
            
        # if 'output' in data[i]:
        #     rewrite = data[i]['output']
        # elif 'rewrite' in data[i]:
        #     rewrite = data[i]['rewrite']
        if 'oracle_utt_text' in data[i]:
            rewrite = data[i]['oracle_utt_text']
        else: #  'original_oracle_utt_text' in data[i]: # 'original_oracle_utt_text' in data[i]:
            rewrite = data[i]['original_oracle_utt_text']
            
        if 'query' in data[i]:
            cur_query = data[i]['query']
        else:
            cur_query = data[i]['cur_utt_text']
        
        if args.eval_type == "answer":
            data_2[i] = json.loads(data_2[i])
            rewrite = data_2[i]['answer_utt_text']
            
        elif args.eval_type == "human_rewrite":
            rewrite = data[i]['rewrite']
            
        elif args.eval_type == "original":
            rewrite = cur_query
            
        # ! for back-retrieval
        elif args.eval_type == "this_answer_only":
            rewrite = data[i]['answer']
            
        # ! for back-retrieval
        elif args.eval_type ==  "this_answer_query": 
            rewrite = data[i]['query'] # T5-trained rewrite
            rewrite = rewrite + ' ' + data[i]['answer']

        # ! for back-retrieval
        elif args.eval_type ==  "this_answer_truth_rewrite": 
            rewrite = data[i]['original_oracle_utt_text'] # T5-trained rewrite
            rewrite = rewrite + ' ' + data[i]['answer']

        # ! for back-retrieval
        elif args.eval_type.startswith("originalCTX_resp"): 
            bwd_n, fwd_n = args.eval_type.split("_")[1], args.eval_type.split("_")[2]
            bwd_n, fwd_n = int(bwd_n.split("bwd")[-1]), int(fwd_n.split("fwd")[-1])

            # get bwd_ctx
            curr_turn_dt = data[i]
            bwd_ctx = [item for pair in zip(curr_turn_dt['history_query'], curr_turn_dt['history_answer'])
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
            fwd_ctx = [item for pair in zip(fwd_turn_dt['history_query'], fwd_turn_dt['history_answer'])
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

            rewrite = bwd_ctx + [data[i]['answer']] + fwd_ctx
            print(len(rewrite), rewrite)

            rewrite = ' '.join(rewrite)

        # ! for back-retrieval
        elif args.eval_type.startswith("resp"): 
            bwd_n, fwd_n = args.eval_type.split("_")[1], args.eval_type.split("_")[2]
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

            rewrite = bwd_ctx + [data[i]['answer']] + fwd_ctx
            print(len(rewrite), rewrite)

            rewrite = ' '.join(rewrite)

        elif args.eval_type == "oracle+answer":
            data_2[i] = json.loads(data_2[i])
            rewrite = rewrite + ' ' + data_2[i]['answer_utt_text']
        elif args.eval_type == "oracle+nexq":
            data_2[i] = json.loads(data_2[i])
            rewrite = rewrite + ' ' + data_2[i]['next_q_utt_text']
        
        elif args.eval_type == "answer":
            data_2[i] = json.loads(data_2[i])
            rewrite = data_2[i]['answer_utt_text']
        queries += [rewrite]
        qids += [sample_id]
    
    print(queries)

    output_f = args.output_f if args.output_f is not None else f"{args.split}_{args.read_by}_{out_sfx}"
    # output_trec_file = oj(args.qrel_output_path, output_f)
    ########## ! imported codes ##########


    # query embeddings
    logger.info("Generating query embeddings for testing...")

    # print(f'Total number of queries: {len(raw_examples)}')
    
    # qids = []
    # queries = []
    # for idx, line in enumerate(raw_examples):
    #     qid, q = line
    #     if q:
    #         qids.append(qid)
    #         queries.append(q)
    print(f'Number of valid queries: {len(queries)}')
    
    # query encoder
    # model = SentenceTransformer('sentence-transformers/gtr-t5-base')
    model = SentenceTransformer(args.model_name_or_path)
    
    model.max_seq_length = 384

    # query embeddings
    embeddings = model.encode(queries, 
                              batch_size=256, 
                              show_progress_bar=True) 
    
    all_scores_list = []
    
    for spt in range(args.num_splits):
        all_scores = {}
        for idx, line in enumerate(zip(qids, queries)):
            qid, q = line
            if not q:
                all_scores[qid] = {}
                continue
                
        # load passage ids
        pids_path = os.path.join(args.dense_index_path, f"doc_ids_{spt}.json")
        pids = json.load(open(pids_path, "r"))
        logger.info(f"Load {len(pids)} pids from {pids_path}")
            
        # load faiss index
        index_path = os.path.join(args.dense_index_path, f"index_test_{spt}.faiss")
        logger.info(f"Load index from {index_path}")
        indexer = DenseIndexer(dim=768,logger=logger)
        indexer.load_index(index_path)
        logger.info(f"Index loading success!!")
    
        scores = indexer.retrieve(embeddings, qids, pids)
    
        all_scores.update(scores)
        
        logger.info(f"Dense search finished")
        
        all_scores_list.append(all_scores)
    
        # json.dump(
        #     all_scores,
        #     open(os.path.join(args.preprocessed_data_path, f"{output_f}_dpr_{spt}.json"), "w"),
        #     indent=4
        # )
    
    merged_results = merge_scores(all_scores_list, 100)
    json.dump(
            merged_results,
            open(os.path.join(args.preprocessed_data_path, f"{output_f}_dpr.json"), "w"),
            indent=4
        )

    