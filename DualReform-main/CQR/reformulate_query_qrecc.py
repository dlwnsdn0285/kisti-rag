import json
import os
# os.environ["TRANSFORMERS_CACHE"] = "/data/../llm_cache"
# import openai
import jsonlines
from tqdm import tqdm
import time
import argparse

# longlora: inference-qlora.py
import sys
# Add the directory containing the target file to sys.path
sys.path.append('/home/../Rag/LongLoRA/')
import math
import torch
import textwrap
import transformers
from peft import PeftModel
from transformers import GenerationConfig, TextStreamer, BitsAndBytesConfig
from llama_attn_replace import replace_llama_attn

# model_name = "gpt-3.5-turbo-0125"
# openai.api_key = ''

PROMPT_DICT = {
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_input_llama2": (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
    "prompt_llama2": "[INST]{instruction}[/INST]"
}


def set_prompt(line, args, n_recent=3, tokenizer=None):
    # (w pssg, wo pssg) x (icl, zsl): prompt_type, use_pssg
    
    # Inst: 
    # "Given a question and its context, decontextualize the question by addressing coreference and omission issues. 
    # The resulting question should retain its original meaning and be as informative as possible, 
    # and should not duplicate any previously asked questions in the context."
    # if args.use_pssg:    
    #     Instruction = "Given a question, its previous questions (Q) & answers (A) and retrieved documents (Document), decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the context."
    # else:
    #     Instruction = "Given a question and its context, decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the context."
    
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
    if args.use_pssg: # using {}-passages-per-line.json
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
        for idx, sent in enumerate(line['NewContext']):
            if idx % 2 == 0:
                curr_ctx.append(f"Q: {sent}")
            else:
                curr_ctx.append(f"A: {sent}")
                
    curr_ctx = " ".join(curr_ctx)
    curr_ctx = f"[{curr_ctx}]"
    # if args.prompt_type == "icl":
    #     if args.use_pssg:
    #         e1 = "Context: [Q: When was Born to Fly released? Document: Born to Fly is a song co-written and recorded by American country music artist Sara Evans. It was released in June 2000 as the first single and title track from her 2000 album of the same name. A: Sara Evans's third studio album, Born to Fly, was released on October 10, 2000.] \nQuestion: Was Born to Fly well received by critics?\nRewrite: Was Born to Fly well received by critics?"
    #         e2 = "Context: [Q: When was Keith Carradine born? Document: No relevant documents. A: Keith Ian Carradine was born August 8, 1949. Q: Is he married? Document: Carradine married Sandra Will on February 6, 1982. They were separated in 1993, before Will filed for divorce in 1999. The couple had two children: Cade Richmond Carradine (born July 19, 1982) and Sorel Johannah Carradine (born June 18, 1985). A: Keith Carradine married Sandra Will on February 6, 1982.]\nQuestion: Do they have any children?\nRewrite: Do Keith Carradine and Sandra Will have any children?"
    #         e3 = "Context: [Q: Who proposed that atoms are the basic units of matter? Document: Arguably the most important of all Dalton's investigations are concerned with the atomic theory in chemistry. While his name is inseparably associated with this theory, the origin of Dalton's atomic theory is not fully understood.[19][20] The theory may have been suggested to him either by researches on ethylene (olefiant gas) and methane (carburetted hydrogen) or by analysis of nitrous oxide (protoxide of azote) and nitrogen dioxide (deutoxide of azote), both views resting on the authority of Thomas Thomson. A: John Dalton proposed that each chemical element is composed of atoms of a single, unique type, and they can combine to form more complex structures called chemical compounds.] \nQuestion: How did the proposal come about?\nRewrite: How did John Dalton's proposal that each chemical element is composed of atoms of a single unique type, and they can combine to form more complex structures called chemical compounds come about?"
    #         e4 = "Context: [Q: What is it called when two liquids separate? Document: Decantation is a process for the separation of mixtures of immiscible liquids or of a liquid and a solid mixture such as a suspension.[1] The layer closer to the top of the container—the less dense of the two liquids, or the liquid from which the precipitate or sediment has settled out—is poured off, leaving denser liquid or the solid behind. The process typically is unable to remove all of the top layer, meaning the separation is incomplete or at least one of the two separated components is still contaminated by the other one. A: Decantation is a process for the separation of mixtures of immiscible liquids or of a liquid and a solid mixture such as a suspension.  Q: How does the separation occur?  Document: No relevant documents.  A: The layer closer to the top of the container-the less dense of the two liquids, or the liquid from which the precipitate or sediment has settled out-is poured off.]\nQuestion: Then what happens?\nRewrite: Then what happens after the layer closer to the top of the container is poured off with decantation?"
        
    #     else: # without past passages    
    #         e1 = "Context: [Q: When was Born to Fly released? A: Sara Evans's third studio album, Born to Fly, was released on October 10, 2000.]\nQuestion: Was Born to Fly well received by critics?\nRewrite: Was Born to Fly well received by critics?"
    #         e2 = "Context: [Q: When was Keith Carradine born? A: Keith Ian Carradine was born August 8, 1949. Q: Is he married? A: Keith Carradine married Sandra Will on February 6, 1982.]\nQuestion: Do they have any children?\nRewrite: Do Keith Carradine and Sandra Will have any children?"
    #         e3 = "Context: [Q: Who proposed that atoms are the basic units of matter? A: John Dalton proposed that each chemical element is composed of atoms of a single, unique type, and they can combine to form more complex structures called chemical compounds.]\nQuestion: How did the proposal come about?\nRewrite: How did John Dalton's proposal that each chemical element is composed of atoms of a single unique type, and they can combine to form more complex structures called chemical compounds come about?"
    #         e4 = "Context: [Q: What is it called when two liquids separate? A: Decantation is a process for the separation of mixtures of immiscible liquids or of a liquid and a solid mixture such as a suspension. Q: How does the separation occur? A: The layer closer to the top of the container-the less dense of the two liquids, or the liquid from which the precipitate or sediment has settled out-is poured off.]\nQuestion: Then what happens?\nRewrite: Then what happens after the layer closer to the top of the container is poured off with decantation?"

    #     prompt = f"{Instruction}\n\n{e1}\n\n{e2}\n\n{e3}\n\n{e4}\n\nContext: {curr_ctx}\nQuestion: {line['Question']}\nRewrite: "
        
    if args.prompt_type == "icl":
        if args.use_pssg:
            # e1 = "Context: [Q: When was Born to Fly released? Document: Born to Fly is a song co-written and recorded by American country music artist Sara Evans. It was released in June 2000 as the first single and title track from her 2000 album of the same name. A: Sara Evans's third studio album, Born to Fly, was released on October 10, 2000.] \nQuestion: Was Born to Fly well received by critics?\nRewrite: Was Born to Fly well received by critics?"
            # e2 = "Context: [Q: When was Keith Carradine born? Document: No relevant documents. A: Keith Ian Carradine was born August 8, 1949. Q: Is he married? Document: Carradine married Sandra Will on February 6, 1982. They were separated in 1993, before Will filed for divorce in 1999. The couple had two children: Cade Richmond Carradine (born July 19, 1982) and Sorel Johannah Carradine (born June 18, 1985). A: Keith Carradine married Sandra Will on February 6, 1982.]\nQuestion: Do they have any children?\nRewrite: Do Keith Carradine and Sandra Will have any children?"
            # e3 = "Context: [Q: Who proposed that atoms are the basic units of matter? Document: Arguably the most important of all Dalton's investigations are concerned with the atomic theory in chemistry. While his name is inseparably associated with this theory, the origin of Dalton's atomic theory is not fully understood.[19][20] The theory may have been suggested to him either by researches on ethylene (olefiant gas) and methane (carburetted hydrogen) or by analysis of nitrous oxide (protoxide of azote) and nitrogen dioxide (deutoxide of azote), both views resting on the authority of Thomas Thomson. A: John Dalton proposed that each chemical element is composed of atoms of a single, unique type, and they can combine to form more complex structures called chemical compounds.] \nQuestion: How did the proposal come about?\nRewrite: How did John Dalton's proposal that each chemical element is composed of atoms of a single unique type, and they can combine to form more complex structures called chemical compounds come about?"
            # e4 = "Context: [Q: What is it called when two liquids separate? Document: Decantation is a process for the separation of mixtures of immiscible liquids or of a liquid and a solid mixture such as a suspension.[1] The layer closer to the top of the container—the less dense of the two liquids, or the liquid from which the precipitate or sediment has settled out—is poured off, leaving denser liquid or the solid behind. The process typically is unable to remove all of the top layer, meaning the separation is incomplete or at least one of the two separated components is still contaminated by the other one. A: Decantation is a process for the separation of mixtures of immiscible liquids or of a liquid and a solid mixture such as a suspension.  Q: How does the separation occur?  Document: No relevant documents.  A: The layer closer to the top of the container-the less dense of the two liquids, or the liquid from which the precipitate or sediment has settled out-is poured off.]\nQuestion: Then what happens?\nRewrite: Then what happens after the layer closer to the top of the container is poured off with decantation?"
            # # e4 = "Context: [No previous conversation.]\nQuestion: Then what happens?\nRewrite: Then what happens after the layer closer to the top of the container is poured off with decantation?"
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
                         

        # prompt = f"{Instruction}\n\nExample 1:\n{e1}\n\nExample 2:\n{e2}\n\nExample 3:\n{e3}\n\nExample 4:\n{e4}\n\nContext: {curr_ctx}\nQuestion: {line['Question']}\nRewrite: "
        prompt = f"{Instruction}\n\n{e1}\n\n{e2}\n\n{e3}\n\n{e4}\n\nContext: {curr_ctx}\nQuestion: {line['Question']}\nRewrite: "
        
        
    elif args.prompt_type == "zsl":
        prompt = f"{Instruction}\n\nContext: {curr_ctx}\nQuestion: {line['Question']}\nRewrite: "

    if args.use_llama3_1:
        prompt = [{'content':prompt, 'role':'user'}]
        prompt = apply_chat_template_content(prompt, tokenizer)
    else: # use llama2+longlora
        prompt_no_input = PROMPT_DICT["prompt_llama2"]
        prompt = prompt_no_input.format_map({"instruction": prompt})
    # print("prompt: ", prompt)
    return prompt

def apply_chat_template_content(prompt, tokenizer):
    messages = prompt
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    add = "<|start_header_id|>assistant<|end_header_id|>"
    return tokenizer.apply_chat_template(messages, tokenize=False) + add


def generate_rewrite(line, respond, args, tokenizer=None):
    prompt = set_prompt(line, args, tokenizer=tokenizer)

    output = respond(prompt=prompt)
    return output


def build_lamma(args):
    if args.use_llama3_1:
        response_template = "Rewrite:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
)
        
        if args.sft_model: 
            model_id = args.sft_model
            # Load the base model
            model = transformers.AutoModelForCausalLM.from_pretrained(model_id,
                                                        torch_dtype= "auto",
                                                        device_map="auto", 
                                                        quantization_config=quantization_config
                                                        )
        else: 
            model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            # Load the base model
            llama_cache_dir = "/data/../llm_cache"
            model = transformers.AutoModelForCausalLM.from_pretrained(model_id,
                                                        torch_dtype= "auto",
                                                        device_map="auto", 
                                                        quantization_config=quantization_config,
                                                        cache_dir = llama_cache_dir                                                    
                                                        )
    
            # os.environ["TRANSFORMERS_CACHE"] = "/data/../llm_cache"

        # Base model and adapter model names
        adapter_model_name = args.base_model
        # "/data2/../nlp_data/qrecc_llama3_1/Llama-3.1-8B-sft-lora-ignoreindex-best13k/checkpoint-570" 
        
        # Load the adapter model
        model = PeftModel.from_pretrained(model, adapter_model_name)

        # Load the tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        # set pad_token_id equal to the eos_token_id if not set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Set reasonable default for models without max length
        if tokenizer.model_max_length > 100_000:
            tokenizer.model_max_length = 4096 # 16384 # 2048


        # Define special tokens
        special_tokens_dict = {
            "additional_special_tokens": [response_template],
        }

        # Add the special tokens to the tokenizer
        tokenizer.add_special_tokens(special_tokens_dict)

        # Resize the model embeddings to accommodate the new special tokens
        model.resize_token_embeddings(len(tokenizer))

        model.eval()
        
        respond = build_generator_llama3_1(model, tokenizer, temperature=args.temperature,
                            top_p=args.top_p, max_gen_len=args.max_gen_len, use_cache=True)
        return respond, tokenizer
    
    else: # ! llama2+longlora 
        if args.flash_attn:
            replace_llama_attn(inference=True)

        if args.sft_model: 
            model_id = args.sft_model
            
            # Set RoPE scaling factor
            config = transformers.AutoConfig.from_pretrained(
                model_id,
                cache_dir=args.cache_dir,
            )

            orig_ctx_len = getattr(config, "max_position_embeddings", None)
            if orig_ctx_len and args.context_size > orig_ctx_len:
                scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
                config.rope_scaling = {"type": "linear", "factor": scaling_factor}

            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                cache_dir=args.cache_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            )
            adapter_model_name = args.base_model
            model = PeftModel.from_pretrained(model, adapter_model_name)

        else:
            # Set RoPE scaling factor
            config = transformers.AutoConfig.from_pretrained(
                args.base_model,
                cache_dir=args.cache_dir,
            )

            orig_ctx_len = getattr(config, "max_position_embeddings", None)
            if orig_ctx_len and args.context_size > orig_ctx_len:
                scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
                config.rope_scaling = {"type": "linear", "factor": scaling_factor}

            # Load model and tokenizer
            model = transformers.AutoModelForCausalLM.from_pretrained(
                args.base_model,
                config=config,
                cache_dir=args.cache_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            )
            

        model.resize_token_embeddings(32001)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
            padding_side="right",
            use_fast=False,
        )

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        respond = build_generator(model, tokenizer, temperature=args.temperature, top_p=args.top_p,
                                max_gen_len=args.max_gen_len, use_cache=True)
        return respond

def build_generator(
    model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True
):
    def response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        streamer = TextStreamer(tokenizer)
        
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            streamer=streamer,
        )
        
        out = tokenizer.decode(output[0], skip_special_tokens=True)

        out = out.split(prompt.lstrip("<s>"))[1].strip()
        return out

    return response

def build_generator_llama3_1(
    model, tokenizer,  temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True
):
    def response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        streamer = TextStreamer(tokenizer)
        
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            streamer=streamer,
        )
        
        out = tokenizer.decode(output[0],)
        response_template = tokenizer.special_tokens_map['additional_special_tokens'][0]
        out = remove_special_tokens(out.split(response_template)[1].strip(), tokenizer)
        # out = out.split(prompt.lstrip("<s>"))[1].strip()

        return out

    return response

def remove_special_tokens(text, tokenizer):
    for token in tokenizer.special_tokens_map.values():
        if isinstance(token, list):
            token = token[0]
        text = text.replace(token, "")
    # for token in tokenizer.additional_special_tokens:
    #     text = text.replace(token, "")
    return text.strip()

if __name__ == "__main__":
    # args setup #######
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, default=None) # e.g., 'test-modified-sampled.json'
    parser.add_argument('--output_path', type=str, default=None) # e.g., 'test-modified-sampled.json'
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--prompt_type', type=str, default='zsl') # e.g., 'icl', 'zsl', etc.
    parser.add_argument("--use_pssg", action="store_true", help="use pssg information")
    parser.add_argument("--no_save", action="store_true", help="no_save rewrites")
    parser.add_argument("--use_llama3_1", action="store_true", help="use_llama3_1")
    parser.add_argument('--instruct_pssg', type=str, default="original") # original, filter_irrelevant, summary, filter_irrelevant_summary
    # longlora: args    
    # parser.add_argument('--material', type=str, default="")
    # parser.add_argument('--question', type=str, default="")
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--sft_model', type=str, default="")
    parser.add_argument('--cache_dir', type=str, default=None) # ./cache
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--temperature', type=float, default=0.6, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--max_gen_len', type=int, default=512, help='')
    
    args = parser.parse_args()
    print(args)
    ####################
    split = 'test'
    root = 'datasets/qrecc/' if args.root is None else args.root
    if args.input_data is not None:
        lines = json.load(open(os.path.join(root, args.input_data), "r", encoding="utf-8")) 
        # e.g., 'test-modified-sampled.json'
        split = args.input_data.split('-')[0] # e.g., 'test'
    else:
        lines = json.load(open(os.path.join(root, f'{split}-modified.json'), "r", encoding="utf-8"))
    
    from random import sample
    lines = sample(lines, 400)
    
    if split == "test" or split == "dev":
        qrels = json.load(open(os.path.join(root, f"qrels_{split}.txt"), "r"))
    
    if split == "train":
        with open(os.path.join(root, f"qrels_{split}.json"), 'r') as file:
            qrels = json.load(file)
    # print(split)

    if args.use_llama3_1:
        respond, tokenizer = build_lamma(args)
    else:
        respond = build_lamma(args)
        tokenizer = None



    out_path = args.output_path if args.output_path is not None else f'{split}_chatgpt_ZSL_...jsonl'
    with jsonlines.open(os.path.join(root, out_path), mode='a') as writer:
        for line in tqdm(lines):
            conv_id = f"{line['Conversation_no']}_{line['Turn_no']}"
            no_rels = False
            if split == "test" or split == "dev":
                # print(list(qrels[conv_id].keys()))
                if list(qrels[conv_id].keys())[0] == '':
                    no_rels = True
                    # print(conv_id)
            if split == "train":
                if len(list(qrels[conv_id].keys()))==0:
                    no_rels = True
            if no_rels:
                continue
            
            if len(line['NewContext']) == 0:
                line['GPT_rewrite'] = line['Truth_rewrite']
            else:
                line['GPT_rewrite'] = generate_rewrite(line, respond, args, tokenizer=tokenizer)
#             del line['Context']

            # post-processing
            # if line['Turn_no'] > 1:
            #     if args.instruct_pssg.find('summary') > -1 or args.instruct_pssg.find('reasoning') > -1:
            #         # print("################ bef Rewritten Query: ", line['GPT_rewrite']['choices'][0]['message']['content'])
            #         preds = line['GPT_rewrite']['choices'][0]['message']['content'].split('The rewritten query is')
            #         # if len(preds)== 1:
            #         #     preds = line['GPT_rewrite']['choices'][0]['message']['content'].split('The rewritten question is')
            #         # if len(preds)== 1:
            #         #     preds = line['GPT_rewrite']['choices'][0]['message']['content'].split('The query is')
            #         # # debug: "The rewritten question is" "The query is" "revised query is"; ignore not-question queries
            #         pred = preds[-1] if len(preds)>1 else line['GPT_rewrite']['choices'][0]['message']['content']
            #         pred = pred.replace("\"", "")
            #         # if pred[-1] != "?": # not-question query
            #         #     pred = pred + "?"
                        
            #         line['GPT_rewrite']['choices'][0]['message']['content'] = pred
                    
            #         print("################ aft Rewritten Query: ", line['GPT_rewrite']['choices'][0]['message']['content'])

            if args.no_save:
                continue
            
            writer.write(line)
    
    # Pass fuse_rewrite.py & post_processing.py
    lines = []
    with jsonlines.open(os.path.join(root, out_path)) as reader:
        for line in reader:
            lines.append(line)

    # out_file = args.rw_file[args.rw_file.find("chatgpt_"):].strip(".jsonl").lstrip("chatgpt_")
    # if args.suffix is not None:
    #     out_file = out_file + "_" + args.suffix
    # with open(os.path.join(args.root_path, f'{split}_fused_{out_file}.json'), 'w') as out:
    #     json.dump(new_lines, out, indent=2)
    json_out_path = "".join(out_path.split(".jsonl")[:-1]+['.json'] )
    with open(os.path.join(root, json_out_path), 'w') as out:
        json.dump(lines, out, indent=2)

    try:
        os.remove(os.path.join(root, out_path))
    except OSError:
        pass
