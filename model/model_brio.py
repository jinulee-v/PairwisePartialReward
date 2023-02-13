import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

from .pibleu import get_pibleu_score
from .model import ParaphraserBase

class Paraphraser(ParaphraserBase):
    """
    Implementation of BRIO(Bringing Order to Abstractive Summarization) for diverse paraphrase generation
    """

    def __init__(self,
            base: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            num_beams: int = None,
            contrast_lambda : float = None,
            len_penalty: float = None,
            device: torch.device = torch.device("cpu"), **kwargs):
        super(Paraphraser, self).__init__(base, tokenizer, num_beams=num_beams, device=device)

        # BART Layer
        self.base = base
        self.tokenizer = tokenizer
        self.pad_id = self.base.config.pad_token_id
        self.len_penalty = len_penalty

        self.num_beams = num_beams
        self.contrast_lambda = contrast_lambda
        self.device = device


    def get_contrastive_loss(self, inputs, outputs):
        """
        Calculates the token_wise contrastive loss.
        @param inputs List[str]
        @param outputs List[str]

        @return loss
        """
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        batch_size = len(inputs)

        # Tokenize
        input_ids = self.tokenizer(inputs, truncation=True)["input_ids"]
        input_ids = [torch.tensor(idx, device=self.device) for idx in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        attention_mask = input_ids != self.pad_id

        with torch.no_grad():
            # Generate in beam sequences(beam size = batch size)
            output = self.base.generate(
                input_ids,
                num_beams=self.num_beams,
                # Output control
                # max_new_tokens=int(input_ids.size(1)),
                num_return_sequences=batch_size,
                return_dict_in_generate=True,
                output_scores=True,
                early_stopping=True
            )
            sequences = output.sequences.reshape(batch_size, self.num_beams, -1)[:, :, 1:]
            decoder_mask = sequences != self.pad_id

            # Rank the outputs
            pibleu_score = get_pibleu_score(input_ids, sequences, self.tokenizer) # batch_size * num_beams
            ranks = torch.argsort(pibleu_score, dim=1).to(torch.float32)

            # Generate sequence pair differences
            rank_diff_matrix = ranks.unsqueeze(2) - ranks.unsqueeze(1) # batch_size * num_beams * num_beams
            rank_diff_matrix *= self.contrast_lambda
            rank_diff_mask = (rank_diff_matrix > 0).to(torch.float32)

        # Compare beams according to their rank
        # we compute single input and its output beams one by one(that's why we set beam_size to batch_size)
        contrast_loss = 0

        # Retrieve logits and take log-softmax
        contrast_loss = 0
        for i in range(batch_size):
            logits = self.base(
                    input_ids=torch.tile(input_ids[i].unsqueeze(0), (self.num_beams, 1)),
                    attention_mask=torch.tile(attention_mask[i].unsqueeze(0), (self.num_beams, 1)),
                    decoder_input_ids=sequences[i],
                    decoder_attention_mask=decoder_mask[i]
            ).logits # num_beams * seq_len * vocab_size

            # Calculate NLL losses and length penalty
            losses = - loss_fct(logits.reshape(-1, logits.size(2)), sequences[i].reshape(-1))
            losses = losses.reshape(logits.size(0), logits.size(1)) * decoder_mask[i] # num_beams * seq_len
            losses = torch.sum(losses, dim=-1) / torch.pow(torch.sum(decoder_mask[i], dim=1) - 1, self.len_penalty)
            
            # calculate pairwise loss
            loss_diff_matrix = losses.unsqueeze(1) - losses.unsqueeze(0)
            loss_terms = torch.max(torch.zeros_like(loss_diff_matrix), rank_diff_matrix[i] - loss_diff_matrix)
            loss_terms *= rank_diff_mask[i]
            contrast_loss += torch.sum(loss_terms) / torch.sum(rank_diff_mask[i]) # Normalize by (seq1, seq2) combination count
        
        return contrast_loss / batch_size # Normalize by batch size
