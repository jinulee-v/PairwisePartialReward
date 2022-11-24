from argparse import ArgumentParser
import json
from tqdm import tqdm
import os, sys
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from datasets import load_dataset
from transformers import BartForConditionalGeneration, AutoTokenizer

from model.paraconfee import ParaConfee
from model.dataset import ParaphraseGenerationDataset, pg_collate_fn

model_id = "facebook/bart-base"


def main(args):
    torch.manual_seed(args.torch_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Make checkpoint/log directory
    model_store_path = os.path.join(args.model_store_path, args.model_postfix)
    try:
        os.mkdir(model_store_path)
    except FileExistsError:
        if args.secure:
            prompt = input("WARNING: overwriting directory " + model_store_path + ". Continnue? (y/n)")
            if prompt != "y":
                exit()

    # Init logger
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(os.path.join(model_store_path, "train.log"))
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
    bart_model = BartForConditionalGeneration.from_pretrained(model_id)
    bart_tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ParaConfee(
        bart_model,
        bart_tokenizer,
        num_beams=args.num_beams,
        num_beam_groups=args.num_beam_groups,
        from_batch_neg_examples=args.from_batch_neg_examples,
        diversity_penalty=args.diversity_penalty,
        hinge_lambda=args.hinge_lambda,
        device=device
    ).to(device)

    # Load data
    with open(args.train_data, "r", encoding='UTF-8') as file:
        train_data = json.load(file)
    with open(args.dev_data, "r", encoding='UTF-8') as file:
        dev_data = json.load(file)
    train_dataset = ParaphraseGenerationDataset(train_data)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=pg_collate_fn)
    dev_dataset = ParaphraseGenerationDataset(dev_data, shuffle=False)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=pg_collate_fn)

    # Define criteria and optimizer
    def criteria(inputs, outputs):
        loss = 0
        if args.generation_loss_weight > 0:
            loss += args.generation_loss_weight * model.get_generation_loss(inputs, outputs)
        if args.bc_classification_loss_weight > 0 or args.bc_distance_loss_weight > 0:
            loss += model.get_beam_contrast_loss(
                inputs, outputs,
                classification_loss_weight=args.bc_classification_loss_weight,
                distance_loss_weight=args.bc_distance_loss_weight
            )
        if loss == 0:
            raise ValueError("one of *loss_weights should be > 0.")
        return loss

    optimizer = Adam(model.parameters(), lr=args.lr)

    min_loss = 1e+10
    for epoch in range(args.epoch):  # loop over the dataset multiple times
        logger.info(f"< epoch {epoch} >")
        # Train phase
        model.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, outputs = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = criteria(inputs, outputs)
            loss.backward()
            optimizer.step()

            if i % args.log_interval == args.log_interval-1 or i == len(train_loader)-1:
                # Eval phase (on dev set)
                model.eval()
                total = len(dev_data)
                dev_loss = 0
                first_batch=True
                for data in dev_loader:
                    inputs, outputs = data
                    if first_batch:
                        test_input = inputs[0]
                        test_outputs = model.generate([test_input])[0]
                        first_batch=False
                    dev_loss += (criteria(inputs, outputs) * len(inputs)).item()
                logger.info("=================================================")
                logger.info(f"epoch {epoch}, step {i}")
                logger.info(f"dev loss = {dev_loss/total}")
                logger.info("")
                logger.info("Test generation result")
                logger.info(f"input: {test_input}")
                logger.info(f"output:")
                for test_output in test_outputs:
                    logger.info(f"  {test_output}")
                logger.info("")
                if dev_loss/total < min_loss:
                    logger.info(f"Updating min_loss = {min_loss} -> {dev_loss/total}")
                    min_loss = dev_loss / total
                    if args.maintain_best_chkpt_only:
                        os.remove(os.path.join(model_store_path, name))
                    logger.info("Save model checkpoint because reduced loss...")
                    name = f"ParaConfee_{args.model_postfix}_epoch_{epoch}_step_{i+1}.pt"
                    torch.save(model.state_dict(), os.path.join(model_store_path, name))
                logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--dev_data", required=True)

    # Hyperparameters
    parser.add_argument("--torch_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--num_beams", type=int, default=12)
    parser.add_argument("--generation_loss_weight", type=float, default=1.0)
    parser.add_argument("--bc_classification_loss_weight", type=float, default=0.0)
    parser.add_argument("--bc_distance_loss_weight", type=float, default=0.0)
    parser.add_argument("--num_beam_groups", type=int, default=4)
    parser.add_argument("--from_batch_neg_examples", type=int, default=5)
    parser.add_argument("--diversity_penalty", type=float, default=0.9)
    parser.add_argument("--hinge_lambda", type=float, default=0.9)
    parser.add_argument("--log_interval", type=int, default=1000)

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=True)
    parser.add_argument("--model_postfix", required=True)
    parser.add_argument("--maintain_best_chkpt_only", default=False)
    parser.add_argument("--secure", default=False)

    args = parser.parse_args()
    main(args)