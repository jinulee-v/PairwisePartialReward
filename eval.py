from argparse import ArgumentParser
import json
from tqdm import tqdm
import os, sys
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from transformers import BartForConditionalGeneration, AutoTokenizer

import bleu
import bert_score

from model.paraconfee import ParaConfee
from model.dataset import ParaphraseGenerationEvalDataset, pg_collate_fn

model_id = "facebook/bart-base"

MSCOCO_REF_COUNT=4
QQP_REF_COUNT=1

def main(args):
    torch.manual_seed(0)

    # For simplicity, if a directory is given, load the last checkpoint(last name in alphabetical order)
    if args.model_store_path.endswith(".pt"):
        model_store_path = args.model_store_path
    else:
        assert os.path.isdir(args.model_store_path)
        log_path = model_store_path = os.path.join(args.model_store_path, args.model_postfix)
        assert os.path.isdir(model_store_path)
        last_checkpoint = sorted([f for f in os.listdir(model_store_path) if f.endswith(".pt")], reverse=True)[0]
        model_store_path = os.path.join(args.model_store_path, args.model_postfix, last_checkpoint)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Init logger
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(os.path.join(log_path, "eval.log"))
    file_handler.setFormatter(formatter)
    logger = logging.getLogger('')
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # Log basic info
    logger.info("Training arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info("- %s: %r", arg, value)
    logger.info("")

    # Load model
    # Load state_dict and recover non-tensor member variables
    bart = BartForConditionalGeneration.from_pretrained(model_id)
    bart_tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ParaConfee(
        bart,
        bart_tokenizer,
        num_beams=args.num_beams,
        num_beam_groups=args.num_beam_groups,
        diversity_penalty=args.diversity_penalty
    )
    model.load_state_dict(torch.load(model_store_path))
    model.device = device
    model = model.to(device)

    # Load data
    with open(args.test_data, "r", encoding='UTF-8') as file:
        test_data = json.load(file)
    test_dataset = ParaphraseGenerationEvalDataset(test_data)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=pg_collate_fn)
    if 'mscoco' in args.test_data.lower():
        REF_COUNT = MSCOCO_REF_COUNT
    elif 'qqp' in args.test_data.lower():
        REF_COUNT = QQP_REF_COUNT

    # Eval phase (on dev set)
    model.eval()

    new_goldens = [[] for _ in range(REF_COUNT)]
    new_outputs = [[] for _ in range(model.num_beam_groups)]

    first_batch=True
    for data in tqdm(test_loader):
        inputs, goldens = data
        with torch.no_grad():
            outputs = model.generate(inputs)

        # reorganize goldens/outputs to use bleu score
        for output, golden in zip(outputs, goldens):
            if len(golden) != REF_COUNT:
                # Skip if an example has a different number of golden answers
                continue
            for i, out in enumerate(output):
                new_outputs[i].append(out.replace("\n", " ").strip())
            for j, gold in enumerate(golden):
                new_goldens[j].append(gold.replace("\n", " ").strip())

        if first_batch:
            test_input = inputs[0]
            test_outputs = outputs[0]
            test_refs = goldens[0]
            first_batch = False
        
    # *_scores = List_float[model.num_beam_groups]
    bleu_scores = bleu.multi_list_bleu(new_goldens, new_outputs)
    bert_scores = []
    for beam in new_outputs:
        score = 0
        if args.use_bert_score:
            for reference in new_goldens:
                bert_P, bert_R, bert_F1 = bert_score.score(cands=beam, refs=reference, lang='en')
                score += torch.mean(bert_F1)
            score /= len(new_goldens) # = REF_COUNT
        else:
            score = 0
        bert_scores.append(score)

    logger.info("=================================================")
    logger.info("Test generation result")
    logger.info(f"input: {test_input}")
    logger.info(f"output:")
    for test_output in test_outputs:
        logger.info(f"  {test_output}")
    logger.info(f"reference:")
    for test_ref in test_refs:
        logger.info(f"  {test_ref}")
    logger.info("")
    logger.info("BLEU score")
    logger.info(f"multi-BLEU score for best-1 beam: {bleu_scores[0]}")
    logger.info(f"multi-BLEU score for worst({model.num_beam_groups}) beams: {bleu_scores[-1]}")
    logger.info(f"multi-BLEU score max value: {max(bleu_scores)}")
    logger.info(f"multi-BLEU score min value: {min(bleu_scores)}")
    logger.info(f"multi-BLEU score for all {model.num_beam_groups} beams: {sum(bleu_scores)/len(bleu_scores)}")

    logger.info("")
    logger.info("BERT score")
    logger.info(f"BERT score for best-1 beam: {bert_scores[0]}")
    logger.info(f"BERT score for worst({model.num_beam_groups}) beams: {bert_scores[-1]}")
    logger.info(f"BERT score max value: {max(bert_scores)}")
    logger.info(f"BERT score min value: {min(bert_scores)}")
    logger.info(f"BERT score for all {model.num_beam_groups} beams: {sum(bert_scores)/len(bert_scores)}")
    logger.info("")
    logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--test_data", required=True)

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=12)
    parser.add_argument("--num_beam_groups", type=int, default=4)
    parser.add_argument("--diversity_penalty", type=float, default=0.9)

    # Eval stats
    parser.add_argument("--use_bert_score", action="store_true", default=False)

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=True)
    parser.add_argument("--model_postfix", required=True)

    args = parser.parse_args()
    main(args)