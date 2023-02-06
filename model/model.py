import random

import torch
import torch.nn as nn
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
            sample_size: int = None,
            device: torch.device = torch.device("cpu")):
        super(Paraphraser, self).__init__()

        # BART Layer
        self.base = base
        self.tokenizer = tokenizer
        self.pad_id = self.base.config.pad_token_id

        self.num_beams = num_beams
        self.contrast_lambda = contrast_lambda
        self.sample_size = sample_size
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
