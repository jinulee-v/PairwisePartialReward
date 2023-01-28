from argparse import ArgumentParser
import json
from tqdm import tqdm
import os, sys
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BartForConditionalGeneration, AutoTokenizer

from model.model import Paraphraser
from model.dataset import ParaphraseGenerationEvalDataset, pg_collate_fn

model_id = "facebook/bart-base"


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
    file_handler = logging.FileHandler(os.path.join(log_path, "inference.log"))
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
    model.load_state_dict(torch.load(model_store_path))
    model.device = device
    model = model.to(device)

    # Load data
    with open(args.test_data, "r", encoding='UTF-8') as file:
        test_data = json.load(file)
    test_dataset = ParaphraseGenerationEvalDataset(test_data)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=pg_collate_fn)

    # Eval phase (on dev set)
    model.eval()

    result = []
    first_batch=True
    for data in tqdm(test_loader):
        inputs, _ = data
        with torch.no_grad():
            outputs = model.generate(inputs)

        for outputs, reference in zip(outputs, inputs):
            result.append({
                "input": reference,
                "paraphrases": outputs
            })

        if first_batch:
            test_input = inputs[0]
            test_outputs = outputs
            first_batch = False
    
    result_store_path = os.path.join(args.model_store_path, args.model_postfix, "result.json")
    with open(result_store_path, "w", encoding="UTF-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=4)

    logger.info("=================================================")
    logger.info("Test generation result")
    logger.info(f"input: {test_input}")
    logger.info(f"output:")
    for test_output in test_outputs:
        logger.info(f"  {test_output}")
    logger.info("")

    logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--test_data", required=True, help="Test set(JSON file)")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="testing batch size")
    parser.add_argument("--num_beams", type=int, default=12, help="number of beams(generated sequences) per inference")

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=False, default='checkpoints', help="Directory to store model checkpoints.")
    parser.add_argument("--model_postfix", required=True, help="Name for the model.")

    args = parser.parse_args()
    main(args)