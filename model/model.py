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

class Paraphraser(nn.Module):
    """
    BART based module
    """

    def __init__(self,
            base: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            num_beams: int = None,
            contrast_lambda : float = None,
            len_penalty: float = None,
            mix_rate: float = None,
            device: torch.device = torch.device("cpu")):
        super(Paraphraser, self).__init__()

        # BART Layer
        self.base = base
        self.tokenizer = tokenizer
        self.pad_id = self.base.config.pad_token_id
        self.len_penalty = len_penalty
        self.mix_rate = mix_rate

        self.num_beams = num_beams
        self.contrast_lambda = contrast_lambda
        self.device = device

    def get_generation_loss(self, inputs, outputs):
        """
        Calculates classic teacher-forced generation loss.
        @param inputs List[str]
        @param outputs List[str]

        @return loss
        """
        torch.cuda.empty_cache()
        assert len(inputs) == len(outputs)
        batch_size = len(inputs)

        # Tokenize
        input_ids = self.tokenizer(inputs, truncation=True)["input_ids"]
        input_ids = [torch.tensor(idx) for idx in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id).to(self.device)
        attention_mask = input_ids != self.pad_id
        decoder_input_ids = self.tokenizer(outputs, truncation=True)["input_ids"]
        decoder_input_ids = [torch.tensor(idx) for idx in decoder_input_ids]
        decoder_input_ids = pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.pad_id).to(self.device)
        # decoder_attention_mask = decoder_input_ids != self.pad_id

        # Run BART forward pass with teacher forcing
        loss = self.base.forward(
            input_ids,
            attention_mask,
            labels=decoder_input_ids,
            return_dict=True
        ).loss
        
        return loss

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

        # Generate in beam sequences(beam size = batch size)
        output = self.base.generate(
            input_ids,
            num_beams=batch_size,
            # Output control
            num_return_sequences=batch_size,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Rank beams
        with torch.no_grad():
            sequences = output.sequences.reshape(batch_size, batch_size, -1)[:, :, 1:]

            # Rank the outputs
            pibleu_score = get_pibleu_score(input_ids, sequences, self.tokenizer) # batch_size * (batch_size)
            ranks = torch.argsort(pibleu_score, dim=1).to(torch.float32)

            # Generate sequence pair differences
            rank_diff_matrix = F.relu(ranks.unsqueeze(2) - ranks.unsqueeze(1)) # batch_size * (batch_size)  * (batch_size)
            rank_diff_matrix *= self.contrast_lambda
            rank_diff_mask = (rank_diff_matrix != 0).to(torch.float32)

        # Calculate NLL losses and length penalty
        losses = -output.sequences_scores.reshape(batch_size, -1)
        
        # calculate pairwise loss
        loss_diff_matrix = losses.unsqueeze(1) - losses.unsqueeze(2)
        loss_terms = torch.max(torch.zeros_like(loss_diff_matrix), rank_diff_matrix - loss_diff_matrix)
        loss_terms *= rank_diff_mask
        contrast_loss = torch.sum(loss_terms) / torch.sum(rank_diff_mask)
        
        return contrast_loss * self.mix_rate

    
    def generate(self, inputs, skip_special_tokens=True):
        batch_size = len(inputs)

        # Tokenize
        input_ids = self.tokenizer(inputs, truncation=True)["input_ids"]
        input_ids = [torch.tensor(idx) for idx in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id).to(self.device)

        # Run BART generation
        output = self.base.generate(
            input_ids,
            num_beams=self.num_beams,
            # Output control
            max_new_tokens=int(input_ids.size(1) * 1.5),
            num_return_sequences=self.num_beams,
            return_dict_in_generate=True,
            output_scores=True,
        )
        # Convert ids to tokens
        output = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=skip_special_tokens)
        
        # Reshape
        results = []
        i = 0
        for _ in range(batch_size):
            results.append([])
            for __ in range(self.num_beams):
                results[-1].append(output[i])
                i += 1
        return results
