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
    # Init logger
    assert os.path.isdir(args.model_store_path)
    log_path = os.path.join(args.model_store_path, args.model_postfix)

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
        
    # Obtain scores
    pibleu, para, bleu = get_pibleu_score(
        target_inp=reference,
        samples_all=outputs,
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        eval=True
    )

    # How frequently does the model copy the input?
    first_beam = [beam[0] for beam in outputs]
    first_beam_eq_input = sum([(1 if x==y else 0) for x, y in zip(inputs, outputs)])
    first_beam_eq_ratio = first_beam_eq_input / len(inputs) * 100

    logger.info("=================================================")
    logger.info("Analysis result")

    logger.info("Paraphrases(detected by BERT model on QQP)")
    logger.info(f"Total paraphrases rate: {torch.mean(para).item()}")
    logger.info(f"Paraphrases per beam:")
    for beam_id, score in enumerate(torch.mean(para, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")

    logger.info("")
    logger.info("PiBLEU score")
    logger.info(f"Total average: {torch.mean(pibleu)}")
    logger.info(f"PiBLEU score per beam:")
    for beam_id, score in enumerate(torch.mean(pibleu, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    pibleu_sorted, _ = torch.sort(pibleu, dim=-1)
    logger.info(f"PiBLEU score per beam(sorted):")
    for beam_id, score in enumerate(torch.mean(pibleu_sorted, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    
    logger.info("")
    logger.info("Repeated original sentence")
    logger.info("  (in the first beam of model output)")
    logger.info(f"{first_beam_eq_input} / {len(inputs)} ({first_beam_eq_ratio} %)")

    logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=False, default='checkpoints', help="Directory to store model checkpoints.")
    parser.add_argument("--model_postfix", required=True)

    args = parser.parse_args()
    main(args)
