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
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration, AutoTokenizer

# from model.model import Paraphraser -> Paraphraser is imported based on args.loss_fn
from model.dataset import *
from model.pibleu import set_gpu


MODEL_ID = {
    'bart': 'facebook/bart-base',
    't5': 't5-small',
}
MODEL_CLASS = {
    'bart': BartForConditionalGeneration,
    't5': T5ForConditionalGeneration,
}

def main(args):
    # Set torch
    torch.manual_seed(args.torch_seed)

    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_gpu(args.device) # Set GPU for PiBLEU script evaluation

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

    # Load base model(BART, T5, ...)
    model_id = MODEL_ID[args.base_model]
    model_class = MODEL_CLASS[args.base_model]
    base_model = model_class.from_pretrained(model_id)
    base_tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model
    if args.loss_fn == "triecl":
        from model.model_triecl import Paraphraser
    elif args.loss_fn == "brio":
        from model.model_brio import Paraphraser
    elif args.loss_fn == "mrt":
        pass
        # from model.model_mrt import Paraphraser
    else:
        raise ValueError("loss_fn should be in: 'triecl', 'brio', 'mrt'")
    model = Paraphraser(
        base_model,
        base_tokenizer,
        num_beams=args.num_beams,
        contrast_lambda=args.contrast_lambda,
        len_penalty=args.len_penalty,
        device=device
    ).to(device)
    if args.from_checkpoint is not None:
        # Fine-tune from a local checkpoint
        assert os.path.isdir(args.model_store_path)
        model_load_path = os.path.join(args.model_store_path, args.from_checkpoint)
        assert os.path.isdir(model_load_path)
        last_checkpoint = sorted([f for f in os.listdir(model_load_path) if f.endswith(".pt")], reverse=True)[0]
        model_load_path = os.path.join(model_load_path, last_checkpoint)
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        model.device = device
        model = model.to(device)

    # Load data
    with open(args.train_gen_data, "r", encoding='UTF-8') as file:
        train_data = json.load(file)
    with open(args.dev_gen_data, "r", encoding='UTF-8') as file:
        dev_data = json.load(file)
    train_gen_dataset = ParaphraseGenerationDataset(train_data)
    train_gen_loader = DataLoader(train_gen_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=pg_collate_fn, pin_memory=True)
    dev_gen_dataset = ParaphraseGenerationDataset(dev_data, shuffle=False)
    dev_gen_loader = DataLoader(dev_gen_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=pg_collate_fn, pin_memory=True)

    # Define criteria and optimizer
    def criteria(gen_inputs, gen_outputs, debug=False):
        # NLL loss from the decoder head
        loss = 0
        if args.generative:
            new_loss = model.get_generation_loss(gen_inputs, gen_outputs)
            loss += new_loss
            if debug:
                logger.info(f"NLL loss = {new_loss}")

        # Contrast learning in fine-tune state
        if args.contrastive:
            new_loss= model.get_contrastive_loss(gen_inputs, gen_outputs)
            loss += new_loss * args.mix_rate # Multiply mix_rate for weighted sum
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
                            dev_loss += (criteria(gen_inputs, gen_outputs, debug=True)).item() * args.batch_size
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
                    name = f"Model_{args.model_postfix}_epoch_{epoch}_step_{i+1}.pt"
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
    parser.add_argument("--train_gen_data", required=True, help="Training set(JSON file)")
    parser.add_argument("--dev_gen_data", required=True, help="Validation set(JSON file)")

    parser.add_argument("--generative", required=False, action="store_true", help="Use Generative NLL loss for training.")
    parser.add_argument("--contrastive", required=False, action="store_true", help="Use TrieCL contrastive loss for training.")

    # Base model/checkpoint configuration
    parser.add_argument("--from_checkpoint", required=False, default=None, help="Pretrained checkpoint to load and resume training.")
    parser.add_argument("--base_model", required=False, default="bart", choices=["bart", "t5"], help="Base model to train. If using `from_checkpoint`, you do not need to specify this option.")

    # Training objective
    parser.add_argument("--loss_fn", required=False, default="triecl", choices=["triecl", "brio", "mrt"], help="Loss function to use. TrieCL, BRIO, MRT(Minimum Risk Training) are supported")
    parser.add_argument("--offline_dataset", type=str, required=False, help="Whether to use online train or not. Only used for loss_fn='triecl'|'brio'")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: Adam optimizer)")
    parser.add_argument("--epoch", type=int, default=5, help="epoch count")
    parser.add_argument("--num_beams", type=int, default=16, help="number of beams(generated sequences) per inference/contrastive learning")
    parser.add_argument("--contrast_lambda", type=float, default=float('inf'), help="Contrast hinge value (default: triecl==0.5, brio==0.01)")
    parser.add_argument("--len_penalty", type=float, default=1, help="Length penalty (default: brio==1)")
    parser.add_argument("--mix_rate", type=float, default=1, help="(MLE:Loss=1:mix_rate) mix rate (default: 1)")

    parser.add_argument("--log_interval", type=int, default=1000, help="validating / checkpoint saving interval. Validates at the end of each epoch for default.")
    parser.add_argument("--early_stop", type=int, default=4, help="if valid loss does not decrease for `early_stop` validations, stop training.")

    # PyTorch/CUDA configuration
    parser.add_argument("--gpu", type=int, default=0, help="CUDA index for training")
    parser.add_argument("--torch_seed", type=int, default=0, help="torch_seed() value")

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=False, default='checkpoints', help="Directory to store model checkpoints.")
    parser.add_argument("--model_postfix", required=True, help="Name for the model.")
    parser.add_argument("--maintain_best_chkpt_only", default=False, action="store_true", help="If true, remove all checkpoints except the best validation loss. If false, store every best checkpoints")
    parser.add_argument("--secure", required=False, action="store_true", help="")

    args = parser.parse_args()
    # Set contrast_lambda according to loss_fn
    if args.loss_fn == "triecl":
        args.contrast_lambda = 0.5
    elif args.loss_fn == "brio":
        args.contrast_lambda = 0.01

    assert args.generative or args.contrastive

    main(args)