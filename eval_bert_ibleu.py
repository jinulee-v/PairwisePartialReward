from argparse import ArgumentParser
import json
from tqdm import tqdm
import os, sys
import logging

import torch
from scipy.stats import spearmanr

from transformers import AutoTokenizer

from model.metrics import *

def main(args):
    # Set device
    set_gpu(args.gpu) # Set GPU for BERT-iBLEU script evaluation

    # Init logger
    assert os.path.isdir(args.model_store_path)
    log_path = os.path.join(args.model_store_path, args.model_postfix)

    try:
        eval_postfix = args.eval_file.replace("result", "")
    except:
        eval_postfix = args.eval_file
    eval_postfix = eval_postfix.replace(".json", "")

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    if not args.secure:
        # Remove original log file
        if os.path.exists(os.path.join(log_path, f"eval_BERT-iBLEU{eval_postfix}.log")):
            os.remove(os.path.join(log_path, f"eval_BERT-iBLEU{eval_postfix}.log"))
    file_handler = logging.FileHandler(os.path.join(log_path, f"eval_BERT-iBLEU{eval_postfix}.log"))
    file_handler.setFormatter(formatter)
    logger = logging.getLogger('')
    logger.handlers.clear()
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # Log basic info
    logger.info("Training arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info("- %s: %r", arg, value)
    logger.info("")
    
    # Load generated data
    result_store_path = os.path.join(args.model_store_path, args.model_postfix, args.eval_file)
    with open(result_store_path, "r", encoding="UTF-8") as file:
        result = json.load(file)

    reference = []
    outputs = []
    for r in result:
        reference.append(r["source"])
        outputs.append(r["outputs"])
        
    # Obtain scores
    with torch.no_grad():
        bert_ibleu, bert, bleu = get_bert_ibleu_score(
            reference,
            None,
            outputs,
            eval=True
        )
        bert_ibleu = bert_ibleu.cpu()
        beam_index = torch.arange(bert_ibleu.size(1), 0, -1, device=bert_ibleu.device)
        spearman_rho_sum = 0; rho_count = 0
        for score_set in bert_ibleu:
            rho_count += 1
            spearman_rho_sum += spearmanr(score_set, beam_index).correlation
        spearman_rho = spearman_rho_sum / rho_count

    # How frequently does the model copy the input?
    first_beam = [beam[0] for beam in outputs]
    first_beam_eq_input = sum([(1 if x==y else 0) for x, y in zip(reference, first_beam)])
    first_beam_eq_ratio = first_beam_eq_input / len(reference) * 100

    logger.info("=================================================")
    logger.info("Analysis result")

    logger.info("BERT-score")
    logger.info(f"Total average: {torch.mean(bert).item()}")
    logger.info(f"BERT-score per beam:")
    for beam_id, score in enumerate(torch.mean(bert, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    bert_sorted, _ = torch.sort(bert, dim=-1)
    logger.info(f"BERT-score per beam(sorted):")
    for beam_id, score in enumerate(torch.mean(bert_sorted, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")


    logger.info("")
    logger.info("BLEU score")
    logger.info(f"Total average: {torch.mean(bleu)}")
    logger.info(f"BLEU score per beam:")
    for beam_id, score in enumerate(torch.mean(bleu, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    bleu_sorted, _ = torch.sort(bleu, dim=-1)
    logger.info(f"BLEU score per beam(sorted):")
    for beam_id, score in enumerate(torch.mean(bleu_sorted, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")

    logger.info("")
    logger.info("BERT-iBLEU score")
    logger.info(f"Total average: {torch.mean(bert_ibleu)}")
    logger.info(f"BERT-iBLEU score per beam:")
    for beam_id, score in enumerate(torch.mean(bert_ibleu, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    bert_ibleu_sorted, _ = torch.sort(bert_ibleu, dim=-1)
    logger.info(f"BERT-iBLEU score per beam(sorted):")
    for beam_id, score in enumerate(torch.mean(bert_ibleu_sorted, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    
    logger.info("")
    logger.info(f"BERT-iBLEU score Spearman's rho:")
    logger.info(f"  {spearman_rho}")
    logger.info("Repeated original sentence")
    logger.info("  (in the first beam of model output)")
    logger.info(f"{first_beam_eq_input} / {len(reference)} ({first_beam_eq_ratio} %)")

    logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=False, default='checkpoints', help="Directory to store model checkpoints.")
    parser.add_argument("--model_postfix", required=True)
    parser.add_argument("--eval_file", required=False, default='result.json', help="Name of the result file(generated by inference.py)")
    
    parser.add_argument("--gpu", type=int, default=0, help="CUDA index for training")
    parser.add_argument("--secure", required=False, action="store_true", help="")

    args = parser.parse_args()
    main(args)
