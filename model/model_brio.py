import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

from .model import ParaphraserBase

class Paraphraser(ParaphraserBase):
    """
    Implementation of BRIO(Bringing Order to Abstractive Summarization) for diverse paraphrase generation
    """

    def __init__(self,
            base: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            metric: callable,
            num_beams: int = None,
            contrast_lambda : float = None,
            len_penalty: float = None,
            device: torch.device = torch.device("cpu"), **kwargs):
        super(Paraphraser, self).__init__(base, tokenizer, num_beams=num_beams, device=device)

        # BART Layer
        self.base = base
        self.tokenizer = tokenizer
        self.metric = metric
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
        input_ids_list = [torch.tensor(idx, device=self.device) for idx in input_ids]
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_id)

        with torch.no_grad():
            # Generate in beam sequences(beam size = batch size)
            output = self.base.generate(
                input_ids,
                num_beams=self.num_beams,
                # Output control
                # max_new_tokens=int(input_ids.size(1)),
                num_return_sequences=self.num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                early_stopping=True
            )
            sequences = output.sequences.reshape(batch_size, self.num_beams, -1)
            decoder_mask = sequences != self.pad_id

            # Rank the outputs
            beam_size = sequences.size(1)
            extended_inputs, extended_outputs = [], []
            for in_sent, out_sent in zip(inputs, outputs):
                extended_inputs.extend([in_sent] * beam_size)
                extended_outputs.extend([out_sent] * beam_size)
            samples_str = self.tokenizer.batch_decode(sequences.view(-1, sequences.size(-1)), skip_special_tokens=True) # aggregate batch & sample IDs
            samples_str = [[s] for s in samples_str]
            metrics = self.metric(extended_inputs, extended_outputs, samples_str).reshape(batch_size, beam_size) # batch_size * num_beams
            ranks = torch.argsort(metrics, dim=1).to(torch.float32)

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
            ith_input_ids = torch.tile(input_ids_list[i].unsqueeze(0), (self.num_beams, 1))
            logits = self.base(
                    input_ids=ith_input_ids,
                    attention_mask=(ith_input_ids != self.pad_id),
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
            update_val = torch.sum(loss_terms) / torch.sum(rank_diff_mask[i]) # Normalize by (seq1, seq2) combination count
            if not torch.isnan(update_val): # NaN prevention
                contrast_loss += update_val
        
        return contrast_loss / batch_size # Normalize by batch size
