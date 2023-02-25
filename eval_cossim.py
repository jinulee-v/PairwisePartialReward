from argparse import ArgumentParser
import json
from tqdm import tqdm
import os, sys
import logging

import torch

from sentence_transformers import SentenceTransformer

model_id = "all-mpnet-base-v2"


def main(args):
    # Set device
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Init logger
    assert os.path.isdir(args.model_store_path)
    log_path = os.path.join(args.model_store_path, args.model_postfix)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    if not args.secure:
        # Remove original log file
        if os.path.exists(os.path.join(log_path, "eval_cossim.log")):
            os.remove(os.path.join(log_path, "eval_cossim.log"))
    file_handler = logging.FileHandler(os.path.join(log_path, "eval_cossim.log"))
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

    model = SentenceTransformer(model_id, device=device_str)

    # Obtain scores
    scores = []
    for r in tqdm(result):
        vectors = model.encode([r["input"]] + r["paraphrases"], convert_to_tensor=True, show_progress_bar=False)
        in_vec = vectors[0].unsqueeze(0)
        out_vec = vectors[1:]
        scores.append(torch.nn.functional.cosine_similarity(in_vec, out_vec, dim=1).unsqueeze(0)) # 1 * beam_size
    scores = torch.cat(scores, dim=0) # len(result) * beam_size

    logger.info("=================================================")
    logger.info("Analysis result")
    logger.info("")
    logger.info("Cossine similarity(SentenceTransformer) score")
    logger.info(f"Total average: {torch.mean(scores)}")
    logger.info(f"Cossim score per beam:")
    for beam_id, score in enumerate(torch.mean(scores, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    scores_sorted, _ = torch.sort(scores, dim=-1)
    logger.info(f"Cossim score per beam(sorted):")
    for beam_id, score in enumerate(torch.mean(scores_sorted, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")

    logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=False, default='checkpoints', help="Directory to store model checkpoints.")
    parser.add_argument("--model_postfix", required=True)
    
    parser.add_argument("--gpu", type=int, default=0, help="CUDA index for training")
    parser.add_argument("--secure", required=False, action="store_true", help="")

    args = parser.parse_args()
    main(args)
