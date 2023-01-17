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

from model.model import Paraphraser
from model.dataset import *

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
    model = Paraphraser(
        bart_model,
        bart_tokenizer,
        num_beams=args.num_beams,
        num_beam_groups=args.num_beam_groups,
        diversity_penalty=args.diversity_penalty,
        contrast_lambda=args.contrast_lambda,
        device=device
    ).to(device)
    if args.fine_tune:
        if args.from_checkpoint is not None:
            assert os.path.isdir(args.model_store_path)
            model_load_path = os.path.join(args.model_store_path, args.from_checkpoint)
            assert os.path.isdir(model_load_path)
            last_checkpoint = sorted([f for f in os.listdir(model_load_path) if f.endswith(".pt")], reverse=True)[0]
            model_load_path = os.path.join(model_load_path, last_checkpoint)
            model.load_state_dict(torch.load(model_load_path))
            model.device = device
            model = model.to(device)
        else:
            raise ValueError("To use `fine_tune` arg, `from_checkpoint` must be specified.")

    # Load data
    with open(args.train_gen_data, "r", encoding='UTF-8') as file:
        train_data = json.load(file)
    with open(args.dev_gen_data, "r", encoding='UTF-8') as file:
        dev_data = json.load(file)
    train_gen_dataset = ParaphraseGenerationDataset(train_data)
    train_gen_loader = DataLoader(train_gen_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=pg_collate_fn)
    dev_gen_dataset = ParaphraseGenerationDataset(dev_data, shuffle=False)
    dev_gen_loader = DataLoader(dev_gen_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=pg_collate_fn)

    # Define criteria and optimizer
    def criteria(gen_inputs, gen_outputs, debug=True):
        # NLL loss from the decoder head
        loss = model.get_generation_loss(gen_inputs, gen_outputs)
        if debug:
            logger.info(f"NLL loss = {loss}")

        # Contrast learning in fine-tune state
        if args.fine_tune:
            new_loss= model.get_contrastive_loss(gen_inputs)
            loss += new_loss
            if debug:
                logger.info(f"Contrastive loss = {new_loss}")
        
        if debug:
            logger.info(f"=> Total loss = {loss}")
        return loss

    optimizer = Adam(model.parameters(), lr=args.lr)

    min_loss = 1e+10
    early_stop_count = 0
    for epoch in range(args.epoch):  # loop over the dataset multiple times
        logger.info(f"< epoch {epoch} >")
        # Train phase
        model.train()
        epoch_size = len(train_gen_loader)
        for i, gen_data in enumerate(tqdm(train_gen_loader, total=epoch_size)):
            # get the inputs; data is a list of [inputs, labels]
            gen_inputs, gen_outputs = gen_data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = criteria(gen_inputs, gen_outputs)
            loss.backward()
            optimizer.step()

            if i % args.log_interval == args.log_interval-1 or i == epoch_size-1:
                # Eval phase (on dev set)
                model.eval()
                with torch.no_grad():
                    total = len(dev_data)
                    dev_loss = 0
                    first_batch=True
                    for gen_data in dev_gen_loader:
                        gen_inputs, gen_outputs = gen_data
                        if first_batch:
                            test_input = gen_inputs[0]
                            test_outputs = model.generate([test_input])[0]
                            dev_loss += (criteria(gen_inputs, gen_outputs)).item() * args.batch_size
                            first_batch=False
                        else:
                            dev_loss += (criteria(gen_inputs, gen_outputs)).item() * args.batch_size
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
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    logger.info(f"Min loss not updated for {early_stop_count} validation routines...")
                    if early_stop_count >= args.early_stop:
                        logger.info("Early stopping....")
                        return
                logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--train_gen_data", required=True)
    parser.add_argument("--dev_gen_data", required=True)

    parser.add_argument("--fine_tune", required=False, action="store_true")
    parser.add_argument("--from_checkpoint", required=False, default=None)

    # Hyperparameters
    parser.add_argument("--torch_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--num_beams", type=int, default=12)
    parser.add_argument("--num_beam_groups", type=int, default=12)
    parser.add_argument("--diversity_penalty", type=float, default=0.9)
    parser.add_argument("--contrast_lambda", type=float, default=0.2)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--early_stop", type=int, default=4)

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=True)
    parser.add_argument("--model_postfix", required=True)
    parser.add_argument("--maintain_best_chkpt_only", default=False)
    parser.add_argument("--secure", required=False, action="store_true")

    args = parser.parse_args()
    main(args)