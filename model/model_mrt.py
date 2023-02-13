import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

from .pibleu import get_pibleu_score
from .model import ParaphraserBase

class Paraphraser(ParaphraserBase):
    """
    Implementation of MRT(Minimum Risk Training) for diverse paraphrase generation
    """

    def __init__(self,
            base: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            num_beams: int = None,
            contrast_lambda : float = None,
            sample_size: int = None,
            device: torch.device = torch.device("cpu"), **kwargs):
        super(Paraphraser, self).__init__(base, tokenizer, num_beams=num_beams, device=device)

        # BART Layer
        self.base = base
        self.tokenizer = tokenizer
        self.pad_id = self.base.config.pad_token_id

        self.num_beams = num_beams
        self.contrast_lambda = contrast_lambda
        self.sample_size = sample_size
        self.device = device


    def get_contrastive_loss(self, inputs, outputs):
        """
        Calculates the 'Minimum Risk Training' loss.
        @param inputs List[str]
        @param outputs List[str]

        @return loss
        """
        batch_size = len(inputs)

        # Tokenize
        input_ids = self.tokenizer(inputs, truncation=True)["input_ids"]
        input_ids = [torch.tensor(idx, device=self.device) for idx in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        attention_mask = input_ids != self.pad_id

        # Generate in beam sequences(beam size = batch size)
        output = self.base.generate(
            input_ids,
            num_beams=self.sample_size,
            # Output control
            num_return_sequences=self.sample_size,
            return_dict_in_generate=True,
            do_sample=True,
            output_scores=True
        )

        with torch.no_grad():
            sequences = output.sequences.reshape(batch_size, self.sample_size, -1)[:, :, 1:]

            # Rank the outputs
            pibleu_score = get_pibleu_score(input_ids, sequences, self.tokenizer) # batch_size * sample_size
            
        # Minimum risk training
        token_scores = torch.cat([score.unsqueeze(1) for score in output.scores], dim=1).reshape(batch_size, self.sample_size, len(output.scores), -1)
        token_scores = torch.gather(token_scores, 3, sequences.unsqueeze(3)).squeeze(3)
        token_scores *= (sequences != self.pad_id).to(torch.float32)
        sequence_prob = torch.exp(torch.sum(token_scores, dim=2))
        loss = pibleu_score * sequence_prob / torch.sum(sequence_prob, dim=1, keepdim=True)

        return torch.sum(loss) / (batch_size * self.sample_size)
