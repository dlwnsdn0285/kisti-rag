import json
import os
import openai
import jsonlines
from tqdm import tqdm
import time
import argparse

model_name = "gpt-3.5-turbo-0125"
openai.api_key = ''

def set_prompt(line, args, n_recent=3):
    
    if args.use_pssg:    
        if args.instruct_pssg == 'original':
            Instruction = "Given a question, its previous questions (Q), retrieved documents (Document), and answers (A), decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the context."
        elif args.instruct_pssg == 'filter_irrelevant':
            Instruction = "Given a question, its previous questions (Q), retrieved documents (Document), and answers (A), decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the context. Use the documents to enrich your question if they're relevant, or draw on the Q&A context for a precise reformulation if the documents aren't helpful."
        elif args.instruct_pssg == 'summary':
            Instruction = "Given a question, its previous questions (Q), retrieved documents (Document), and answers (A), decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the context. Given the potential noise and dependencies within the context, creating a concise summary of it first could be an effective strategy for accurately rephrasing the question. Therefore, start by summarizing the context before you decontextualize the question."
        elif args.instruct_pssg == 'filter_irrelevant_summary':
            Instruction = "Given a question, its previous questions (Q), retrieved documents (Document), and answers (A), decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the context. Use the documents to enrich your question if they're relevant, or draw on the Q&A context for a precise reformulation if the documents aren't helpful. Considering the potential noise and dependencies within the context, creating a concise summary of it first could be an effective strategy for accurately rephrasing the question. Therefore, start by summarizing the context before you decontextualize the question."
        elif args.instruct_pssg == 'reasoning':
            Instruction = "Given a question, its previous questions (Q), retrieved documents (Document), and answers (A), decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the context. Use the documents to enrich your question if they're relevant, or draw on the Q&A context for a precise reformulation if the documents aren't helpful."
            Instruction = Instruction + " Before rewriting, evaluate which parts of the context are essential to address, helping to rewrite your question effectively."
    else:
        if args.instruct_pssg == 'original':
            Instruction = "Given a question, its previous questions (Q) and answers (A), decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the context."
        elif args.instruct_pssg == 'filter_irrelevant':
            Instruction = "Given a question, its previous questions (Q) and answers (A), decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the context."
        elif args.instruct_pssg == 'summary':
            Instruction = "Given a question, its previous questions (Q) and answers (A), decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the context. Given the potential noise and dependencies within the context, creating a concise summary of it first could be an effective strategy for accurately rephrasing the question. Therefore, start by summarizing the context before you decontextualize the question."
        elif args.instruct_pssg == 'filter_irrelevant_summary':
            Instruction = "Given a question, its previous questions (Q) and answers (A), decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the context. Considering the potential noise and dependencies within the context, creating a concise summary of it first could be an effective strategy for accurately rephrasing the question. Therefore, start by summarizing the context before you decontextualize the question."
        elif args.instruct_pssg == 'reasoning':
            Instruction = "Given a question, its previous questions (Q) and answers (A), decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the context."
            Instruction = Instruction + " Before rewriting, evaluate which parts of the context are essential to address, helping to rewrite your question effectively."
            
            
    curr_ctx = []
    if args.use_pssg: # ! currently not available
        n_prev_QAturn = len(line['NewContext'])//2
        s_idx_adddocs = max(n_prev_QAturn - n_recent, 0) * 2 # starting-idx to add passage
        p_docs = [ f"Document: {d}." for d in line['Truth_passages_contents'][-n_recent:] ] # recent top1 docs
        
        p_docs_i = 0
        # (Q-Doc-A)-...
        for idx, sent in enumerate(line['NewContext']): # run the below when turn_no >= 1
            if idx % 2 == 0:
                curr_ctx.append(f"Q: {sent}")
                if idx >= s_idx_adddocs:
                    curr_ctx.append(p_docs[p_docs_i])
                    p_docs_i += 1
                else:
                    curr_ctx.append("Document: No relevant documents.")
            else:
                curr_ctx.append(f"A: {sent}")
                
    else:
        if args.ctx_original_qs: # use originial queries in context
            ctx = [ x for pair in zip(line["history_query"], line["history_answer"]) for x in pair]
            for idx, sent in enumerate(ctx):
                if idx % 2 == 0:
                    curr_ctx.append(f"Q: {sent}")
                else:
                    curr_ctx.append(f"A: {sent}")
        
        else:
            ctx = [ x for pair in zip(line["history_rewrite"], line["history_answer"]) for x in pair]
            for idx, sent in enumerate(ctx):
                if idx % 2 == 0:
                    curr_ctx.append(f"Q: {sent}")
                else:
                    curr_ctx.append(f"A: {sent}")
                    
    curr_ctx = " ".join(curr_ctx)
    curr_ctx = f"[{curr_ctx}]"
    
    if args.prompt_type == "icl":
        if args.use_pssg:
            e1 = "Context: [Q: When was Born to Fly released? Document: Born to Fly is a song co-written and recorded by American country music artist Sara Evans. It was released in June 2000 as the first single and title track from her 2000 album of the same name. A: Sara Evans's third studio album, Born to Fly, was released on October 10, 2000.] \nQuestion: Was Born to Fly well received by critics?\nRewrite: Was Born to Fly well received by critics?"
            e2 = "Context: [Q: When was Keith Carradine born? Document: No relevant documents. A: Keith Ian Carradine was born August 8, 1949. Q: Is he married? Document: Carradine married Sandra Will on February 6, 1982. They were separated in 1993, before Will filed for divorce in 1999. The couple had two children: Cade Richmond Carradine (born July 19, 1982) and Sorel Johannah Carradine (born June 18, 1985). A: Keith Carradine married Sandra Will on February 6, 1982.]\nQuestion: Do they have any children?\nRewrite: Do Keith Carradine and Sandra Will have any children?"
            e3 = "Context: [Q: Who proposed that atoms are the basic units of matter? Document: Arguably the most important of all Dalton's investigations are concerned with the atomic theory in chemistry. While his name is inseparably associated with this theory, the origin of Dalton's atomic theory is not fully understood. The theory may have been suggested to him either by researches on ethylene (olefiant gas) and methane (carburetted hydrogen) or by analysis of nitrous oxide (protoxide of azote) and nitrogen dioxide (deutoxide of azote), both views resting on the authority of Thomas Thomson. A: John Dalton proposed that each chemical element is composed of atoms of a single, unique type, and they can combine to form more complex structures called chemical compounds.] \nQuestion: How did the proposal come about?\nRewrite: How did John Dalton's proposal that each chemical element is composed of atoms of a single unique type, and they can combine to form more complex structures called chemical compounds come about?"
            e4 = "Context: [Q: What is it called when two liquids separate? Document: Decantation is a process for the separation of mixtures of immiscible liquids or of a liquid and a solid mixture such as a suspension. The layer closer to the top of the container—the less dense of the two liquids, or the liquid from which the precipitate or sediment has settled out—is poured off, leaving denser liquid or the solid behind. The process typically is unable to remove all of the top layer, meaning the separation is incomplete or at least one of the two separated components is still contaminated by the other one. A: Decantation is a process for the separation of mixtures of immiscible liquids or of a liquid and a solid mixture such as a suspension.  Q: How does the separation occur?  Document: No relevant documents.  A: The layer closer to the top of the container-the less dense of the two liquids, or the liquid from which the precipitate or sediment has settled out-is poured off.]\nQuestion: Then what happens?\nRewrite: Then what happens after the layer closer to the top of the container is poured off with decantation?"
            if args.instruct_pssg == 'original' or args.instruct_pssg == 'filter_irrelevant':
                e1, e2, e3, e4 = e1, e2, e3, e4

            elif args.instruct_pssg == 'summary' or args.instruct_pssg == 'filter_irrelevant_summary':
                e1_tldr = "TLDR Summary: Born to Fly is both a song and the title of Sara Evans's third studio album. The song was released as the album's first single in June 2000, and the album itself was released on October 10, 2000."
                e2_tldr = "TLDR Summary: Keith Ian Carradine, born on August 8, 1949, married Sandra Will on February 6, 1982. They separated in 1993, and Sandra Will filed for divorce in 1999. The couple has two children, Cade Richmond Carradine and Sorel Johannah Carradine."
                e3_tldr = "TLDR Summary: John Dalton proposed the atomic theory, which posits that atoms are the fundamental units of matter, with each chemical element being composed of unique atoms that can combine to form complex compounds. The exact inspiration for Dalton's theory is unclear, but it might have stemmed from his research on gases or the analysis of nitrous oxide and nitrogen dioxide, possibly influenced by Thomas Thomson."
                e4_tldr = "TLDR Summary: The context explains decantation, a separation process for mixtures of immiscible liquids or liquid-solid mixtures like suspensions. It involves pouring off the top, less dense liquid or the liquid cleared of sediment, leaving behind the denser liquid or solid. The process may not completely remove the top layer, potentially leaving some contamination."

                e1 = e1.split('Rewrite:')[0] + 'Rewrite: ' + e1_tldr +\
                         ' The rewritten query is ' + "\"" + e1.split('Rewrite: ')[-1] + "\""
                e2 = e2.split('Rewrite:')[0] + 'Rewrite: ' + e2_tldr +\
                         ' The rewritten query is ' + "\"" + e2.split('Rewrite: ')[-1] + "\""
                e3 = e3.split('Rewrite:')[0] + 'Rewrite: ' + e3_tldr +\
                         ' The rewritten query is ' + "\"" + e3.split('Rewrite: ')[-1] + "\""
                e4 = e4.split('Rewrite:')[0] + 'Rewrite: ' + e4_tldr +\
                         ' The rewritten query is ' + "\"" + e4.split('Rewrite: ')[-1] + "\""

            elif args.instruct_pssg == 'reasoning':
                e1_reasoning = "The question is already clear."
                e2_reasoning = "The original question uses the pronoun \"they\" which is ambiguous without explicit context. By specifying \"Keith Carradine and Sandra Will\" as the subjects, the revised question eliminates any ambiguity about who \"they\" refers to, directly connecting the inquiry to the individuals mentioned in the previous context."
                e3_reasoning = "The original question omits what the proposal actually is. Including the specific details of Dalton's atomic theory (that each chemical element is composed of atoms of a single unique type, and they can combine to form more complex structures called chemical compounds) directly in the question adds necessary context and allows the question to stand alone, making it understandable even without prior knowledge of the conversation."
                e4_reasoning = "The context revolves around decantation, a specific scientific process. Recognizing this as the core topic ensures that the rewrite focuses on the next logical step in this particular procedure. Question: Then what happens? is vague without specifying what it refers to. By identifying that it refers to the action of pouring off the top layer in the decantation process, we address coreference issues, making it clear what the 'then' is referring to."

                e1 = e1.split('Rewrite:')[0] + 'Rewrite: ' + e1_reasoning +\
                         ' The rewritten query is ' + "\"" + e1.split('Rewrite: ')[-1] + "\""
                e2 = e2.split('Rewrite:')[0] + 'Rewrite: ' + e2_reasoning +\
                         ' The rewritten query is ' + "\"" + e2.split('Rewrite: ')[-1] + "\""
                e3 = e3.split('Rewrite:')[0] + 'Rewrite: ' + e3_reasoning +\
                         ' The rewritten query is ' + "\"" + e3.split('Rewrite: ')[-1] + "\""
                e4 = e4.split('Rewrite:')[0] + 'Rewrite: ' + e4_reasoning +\
                         ' The rewritten query is ' + "\"" + e4.split('Rewrite: ')[-1] + "\""

        else: # without past passages    
            
            e1 = "Context: [Q: When was Born to Fly released? A: Sara Evans's third studio album, Born to Fly, was released on October 10, 2000.]\nQuestion: Was Born to Fly well received by critics?\nRewrite: Was Born to Fly well received by critics?"
            e2 = "Context: [Q: When was Keith Carradine born? A: Keith Ian Carradine was born August 8, 1949. Q: Is he married? A: Keith Carradine married Sandra Will on February 6, 1982.]\nQuestion: Do they have any children?\nRewrite: Do Keith Carradine and Sandra Will have any children?"
            e3 = "Context: [Q: Who proposed that atoms are the basic units of matter? A: John Dalton proposed that each chemical element is composed of atoms of a single, unique type, and they can combine to form more complex structures called chemical compounds.]\nQuestion: How did the proposal come about?\nRewrite: How did John Dalton's proposal that each chemical element is composed of atoms of a single unique type, and they can combine to form more complex structures called chemical compounds come about?"
            e4 = "Context: [Q: What is it called when two liquids separate? A: Decantation is a process for the separation of mixtures of immiscible liquids or of a liquid and a solid mixture such as a suspension. Q: How does the separation occur? A: The layer closer to the top of the container-the less dense of the two liquids, or the liquid from which the precipitate or sediment has settled out-is poured off.]\nQuestion: Then what happens?\nRewrite: Then what happens after the layer closer to the top of the container is poured off with decantation?"
            # e4 = "Context: [No previous conversation.]\nQuestion: Then what happens?\nRewrite: Then what happens after the layer closer to the top of the container is poured off with decantation?"
            
            if args.instruct_pssg == 'original' or args.instruct_pssg == 'filter_irrelevant':
                e1, e2, e3, e4 = e1, e2, e3, e4
            
            elif args.instruct_pssg == 'summary' or args.instruct_pssg == 'filter_irrelevant_summary':
                e1_tldr = "TLDR Summary: Inquiry about the release date of Sara Evans's album \"Born to Fly,\" which was on October 10, 2000."
                e2_tldr = "TLDR Summary: Inquiry about Keith Carradine's birth date, which is August 8, 1949, and marital status, revealing he married Sandra Will on February 6, 1982."
                e3_tldr = "TLDR Summary: John Dalton proposed atoms as the basic units of matter, which can combine to form chemical compounds."
                e4_tldr = "TLDR Summary: Decantation separates mixtures of immiscible liquids or liquids and solids by pouring off the top layer after settling."

                e1 = e1.split('Rewrite:')[0] + 'Rewrite: ' + e1_tldr +\
                         ' The rewritten query is ' + "\"" + e1.split('Rewrite: ')[-1] + "\""
                e2 = e2.split('Rewrite:')[0] + 'Rewrite: ' + e2_tldr +\
                         ' The rewritten query is ' + "\"" + e2.split('Rewrite: ')[-1] + "\""
                e3 = e3.split('Rewrite:')[0] + 'Rewrite: ' + e3_tldr +\
                         ' The rewritten query is ' + "\"" + e3.split('Rewrite: ')[-1] + "\""
                e4 = e4.split('Rewrite:')[0] + 'Rewrite: ' + e4_tldr +\
                         ' The rewritten query is ' + "\"" + e4.split('Rewrite: ')[-1] + "\""

            elif args.instruct_pssg == 'reasoning':
                e1_reasoning = "The question is already clear."
                e2_reasoning = "The question \"Do they have any children?\" is ambiguous without directly referencing who \"they\" are. By naming \"Keith Carradine and Sandra Will\" explicitly, we eliminate any ambiguity regarding who the question is about."
                e3_reasoning = "The question \"How did the proposal come about?\" is vague because it doesn't specify which proposal it's referring to. By restating that the proposal is about \"each chemical element being composed of atoms of a single, unique type, and they can combine to form more complex structures called chemical compounds,\" we make the question self-contained."
                e4_reasoning = "The question \"Then what happens?\" is vague without specifying which process it refers to. By stating \"after the layer closer to the top of the container is poured off,\" the question explicitly refers to the action that was previously described, making it clear which stage of the process we're inquiring about what happens next."

                e1 = e1.split('Rewrite:')[0] + 'Rewrite: ' + e1_reasoning +\
                         ' The rewritten query is ' + "\"" + e1.split('Rewrite: ')[-1] + "\""
                e2 = e2.split('Rewrite:')[0] + 'Rewrite: ' + e2_reasoning +\
                         ' The rewritten query is ' + "\"" + e2.split('Rewrite: ')[-1] + "\""
                e3 = e3.split('Rewrite:')[0] + 'Rewrite: ' + e3_reasoning +\
                         ' The rewritten query is ' + "\"" + e3.split('Rewrite: ')[-1] + "\""
                e4 = e4.split('Rewrite:')[0] + 'Rewrite: ' + e4_reasoning +\
                         ' The rewritten query is ' + "\"" + e4.split('Rewrite: ')[-1] + "\""
                         

        prompt = f"{Instruction}\n\n{e1}\n\n{e2}\n\n{e3}\n\n{e4}\n\nContext: {curr_ctx}\nQuestion: {line['query']}\nRewrite: "
        
        
    elif args.prompt_type == "zsl":
        prompt = f"{Instruction}\n\nContext: {curr_ctx}\nQuestion: {line['Question']}\nRewrite: "
        
    # print("prompt: ", prompt)
    return prompt

def generate_rewrite(line, args):
    prompt = set_prompt(line, args)

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    if args.batch_api:
        return messages
    else:
        retries = 5
        delay = 1
        while retries > 0:
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.8, # deterministic decoding: 0.
                    max_tokens=2560,
                    top_p=0.8, # deterministic decoding: 1.
                    n=1, # number of output
                )
                return response
            except:
                pass
            retries -= 1
            time.sleep(delay)
            delay *= 2
        return ""


if __name__ == "__main__":
    # args setup #######
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, default=None) # e.g., 'test-modified-sampled.json'
    parser.add_argument('--output_root', type=str, default=None) # e.g., 
    parser.add_argument('--output_path', type=str, default=None) # e.g., 
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--prompt_type', type=str, default='zsl') # e.g., 'icl', 'zsl', etc.
    parser.add_argument("--use_pssg", action="store_true", help="use pssg information")
    parser.add_argument("--batch_api", action="store_true", help="use pssg information")
    parser.add_argument("--ctx_original_qs", action="store_true", help="contain original qs within ctx")
    parser.add_argument('--instruct_pssg', type=str, default="original") # original, filter_irrelevant, summary, filter_irrelevant_summary
    args = parser.parse_args()
    ####################
    split = 'test'
    root = 'datasets/qrecc/' if args.root is None else args.root
    if args.input_data is not None:
        # lines = json.load(open(os.path.join(root, args.input_data), "r", encoding="utf-8")) 
        with open(os.path.join(root, args.input_data), encoding="utf-8") as f:
            lines = f.readlines()
        lines = [json.loads(l) for l in lines]
        split = args.input_data.split('_')[0] # e.g., 'test'
        
    
    out_path = args.output_path if args.output_path is not None else f'{split}_chatgpt_ZSL_...jsonl'
    output_root = args.output_root if args.output_root is not None else root
    with jsonlines.open(os.path.join(output_root, out_path), mode='a') as writer:
        for line in tqdm(lines):
            conv_id = f"{line['conv_id']}-{line['turn_id']}"
            
            # output attributes
            line['sample_id'] = conv_id
            if 'rewrite' in line:
                line['original_oracle_utt_text'] = line['rewrite']
            line['cur_utt_text'] = line['query']
            if args.batch_api:
                line = generate_rewrite(line, args)
            else:
                line['oracle_utt_text'] = generate_rewrite(line, args)['choices'][0]['message']['content']
            
            writer.write(line)
    
