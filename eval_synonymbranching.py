from argparse import ArgumentParser
import json
from tqdm import tqdm
import os, sys
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BartForConditionalGeneration, AutoTokenizer

from model.model import ParaphraserBase as Paraphraser
from model.pibleu import get_pibleu_score, set_gpu
from model.dataset import SynonymBranchingEvalDataset

model_id = "facebook/bart-base"


def main(args):
    # Set torch
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

    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Set device
    set_gpu(args.gpu) # Set GPU for PiBLEU script evaluation

    # Init logger
    assert os.path.isdir(args.model_store_path)
    log_path = os.path.join(args.model_store_path, args.model_postfix)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    if not args.secure:
        # Remove original log file
        if os.path.exists(os.path.join(log_path, "eval_synonymbranching.log")):
            os.remove(os.path.join(log_path, "eval_synonymbranching.log"))
    file_handler = logging.FileHandler(os.path.join(log_path, "eval_synonymbranching.log"))
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
    model = Paraphraser(
        bart,
        bart_tokenizer,
        num_beams=args.num_beams
    )
    model.load_state_dict(torch.load(model_store_path, map_location=device))
    model.device = device
    model = model.to(device)

    # Load data
    with open(args.test_data, "r", encoding='UTF-8') as file:
        test_data = json.load(file)
    test_dataset = SynonymBranchingEvalDataset(test_data)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    # Eval phase (on dev set)
    model.eval()

    result = []
    for batch in tqdm(test_loader):
        with torch.no_grad():
            result.extend(model.synonym_branching_test(*batch).tolist())
    result = torch.tensor(result)

    logger.info("=================================================")
    logger.info("Analysis result")

    logger.info("")
    logger.info("Synonym branching factor = logp(synonym) - logp(original)")
    logger.info(f"Total average: {torch.mean(result)}")

    logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()

    # Dataset
    parser.add_argument("--test_data", required=True, help="Test set(JSON file)")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="testing batch size")
    parser.add_argument("--num_beams", type=int, default=16, help="number of beams(generated sequences) per inference")

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=False, default='checkpoints', help="Directory to store model checkpoints.")
    parser.add_argument("--model_postfix", required=True)
    
    parser.add_argument("--gpu", type=int, default=0, help="CUDA index for training")
    parser.add_argument("--secure", required=False, action="store_true", help="")

    args = parser.parse_args()
    main(args)
