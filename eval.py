from argparse import ArgumentParser
import json
from tqdm import tqdm
import os, sys
import logging
from tqdm import tqdm

import torch

from transformers import AutoTokenizer

from model.pibleu import get_pibleu_score

model_id = "facebook/bart-base"


def main(args):
    # For simplicity, if a directory is given, load the last checkpoint(last name in alphabetical order)
    if args.model_store_path.endswith(".pt"):
        model_store_path = args.model_store_path
    else:
        assert os.path.isdir(args.model_store_path)
        log_path = model_store_path = os.path.join(args.model_store_path, args.model_postfix)
        assert os.path.isdir(model_store_path)
        last_checkpoint = sorted([f for f in os.listdir(model_store_path) if f.endswith(".pt")], reverse=True)[0]
        model_store_path = os.path.join(args.model_store_path, args.model_postfix, last_checkpoint)

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
    
    # Load generated data
    result_store_path = os.path.join(args.model_store_path, args.model_postfix, "result.json")
    with open(result_store_path, "r", encoding="UTF-8") as file:
        result = json.load(file)

    reference = []
    outputs = []
    for r in result:
        reference.append(r["input"])
        outputs.append(r["paraphrases"])
        
    # *_scores = List_float[model.num_beam_groups]
    pibleu, para, bleu = get_pibleu_score(
        target_inp=reference,
        samples_all=outputs,
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        eval=True
    )

    logger.info("=================================================")
    logger.info("Analysis result")

    logger.info("Paraphrases(detected by BERT model on QQP)")
    logger.info(f"Total paraphrases rate: {torch.mean(para).item()}")
    logger.info(f"Paraphrases per beam:")
    for beam_id, score in enumerate(torch.mean(para, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")

    logger.info("\nPiBLEU score")
    logger.info(f"Total average: {torch.mean(pibleu)}")
    logger.info(f"PiBLEU score per beam:")
    for beam_id, score in enumerate(torch.mean(pibleu, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    pibleu_sorted, _ = torch.sort(pibleu, dim=-1)
    logger.info(f"PiBLEU score per beam(sorted):")
    for beam_id, score in enumerate(torch.mean(pibleu_sorted, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")

    logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=True)
    parser.add_argument("--model_postfix", required=True)

    args = parser.parse_args()
    main(args)