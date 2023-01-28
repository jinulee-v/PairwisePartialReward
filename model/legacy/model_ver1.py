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
            bart: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            num_beams: int = None,
            num_beam_groups: int = None,
            diversity_penalty : float = None,
            contrast_lambda : float = None,
            device: torch.device = torch.device("cpu")):
        super(Paraphraser, self).__init__()

        # BART Layer
        self.bart = bart
        self.tokenizer = tokenizer

        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups
        self.diversity_penalty = diversity_penalty
        self.contrast_lambda = contrast_lambda
        self.device = device

    def get_generation_loss(self, inputs, outputs):
        """
        Calculates classic teacher-forced generation loss.
        @param inputs List[str]
        @param outputs List[str]

        @return loss
        """
        assert len(inputs) == len(outputs)
        batch_size = len(inputs)

        # Tokenize
        input_ids = self.tokenizer(inputs)["input_ids"]
        input_ids = [torch.tensor(idx) for idx in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.bart.config.pad_token_id).to(self.device)
        attention_mask = input_ids != self.bart.config.pad_token_id
        decoder_input_ids = self.tokenizer(outputs)["input_ids"]
        decoder_input_ids = [torch.tensor(idx) for idx in decoder_input_ids]
        decoder_input_ids = pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.bart.config.pad_token_id).to(self.device)
        # decoder_attention_mask = decoder_input_ids != self.bart.config.pad_token_id

        # Run BART forward pass with teacher forcing
        loss = self.bart.forward(
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
        batch_size = len(inputs)

        # Tokenize
        input_ids = self.tokenizer(inputs)["input_ids"]
        input_ids = [torch.tensor(idx) for idx in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.bart.config.pad_token_id).to(self.device)
        attention_mask = input_ids != self.bart.config.pad_token_id

        with torch.no_grad():
            # Generate in beam sequences(beam size = batch size)
            output = self.bart.generate(
                input_ids,
                num_beams=batch_size + 1,
                # Output control
                num_return_sequences=batch_size + 1,
                return_dict_in_generate=True,
                output_scores=True,
            )
            sequences = output.sequences.reshape(batch_size, batch_size+1, -1)[:, :, 1:].tolist()

            # Sort the outputs in lexical order
            sequences = [sorted(sequence) for sequence in sequences] # sort each beams
            sequences = torch.tensor(sequences, device=input_ids.device) # batch_size * beam_size * seq_len

            # Get boundaries and decoder_mask to obtain the shared prefix
            seq1 = sequences[:,  :-1, :]
            seq2 = sequences[:, 1:, :]
            boundaries = (seq1 != seq2).to(torch.long).argmax(dim=2) # batch_size * batch_size
            decoder_mask = torch.tile(torch.arange(0, sequences.size(2)).unsqueeze(0).unsqueeze(1), (boundaries.size(0), boundaries.size(1), 1)).to(boundaries.device)
            decoder_mask = decoder_mask < boundaries.unsqueeze(2)
            assert torch.equal(seq1 * decoder_mask, seq2 * decoder_mask)
            decoder_prefix = seq1 * decoder_mask
            seq1_idx = torch.gather(seq1, 2, boundaries.unsqueeze(2)) # batch_size * batch_size * 1
            seq2_idx = torch.gather(seq2, 2, boundaries.unsqueeze(2))
            first_diff_tok_idx = torch.cat([seq1_idx, seq2_idx], dim=2) # batch_size * batch_size * 2

            # Rank the outputs
            pibleu_score = get_pibleu_score(input_ids, sequences, self.tokenizer) # batch_size * (batch_size+1)
            rank = torch.argsort(pibleu_score, dim=1)
            # tok_dif_sign: 1 if seq1 score is better, else -1
            tok_dif_sign = 2 * (rank[:, :-1] > rank[:, 1:]) - 1 # batch_size * batch_size

        # Compare adjacent beams
        # we compute single input and its output beams one by one(that's why we set beam_size to batch_size)
        contrast_loss = 0
        cnt = 0
        for i in range(batch_size):
            logits = self.bart(
                input_ids=torch.tile(input_ids[i].unsqueeze(0), (batch_size, 1)),
                attention_mask=torch.tile(attention_mask[i].unsqueeze(0), (batch_size, 1)),
                decoder_input_ids=decoder_prefix[i],
                decoder_attention_mask = decoder_mask[i]
            ).logits # batch_size, seq_len, vocab_size
            logits_gather_index = torch.tile(boundaries[i].unsqueeze(1).unsqueeze(2), (1, 1, logits.size(2)))
            logits = torch.gather(logits, 1, logits_gather_index).squeeze(1) # batch_size, vocab_size
            compare_logits = torch.gather(logits, 1, first_diff_tok_idx[i]) # batch_size, 2
            tok_dif = (compare_logits[:, 0] - compare_logits[:, 1]) * tok_dif_sign[i]
            # loss for input = (0 if tok_dif > contrast_lambda ; else contrast_lambda - tok_dif)
            contrast_loss += torch.sum(self.contrast_lambda - torch.min(torch.ones_like(tok_dif) * self.contrast_lambda, tok_dif))
            cnt += tok_dif.size(0)
        
        assert cnt == batch_size ** 2
        return contrast_loss / cnt

    def get_contrastive_loss(self, inputs, outputs):
        """
        Calculates the token_wise contrastive loss.
        @param inputs List[str]
        @param outputs List[str]

        @return loss
        """
        batch_size = len(inputs)
        pad_id = self.bart.config.pad_token_id

        # Tokenize
        input_ids = self.tokenizer(inputs)["input_ids"]
        input_ids = [torch.tensor(idx) for idx in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.bart.config.pad_token_id).to(self.device)
        attention_mask = input_ids != self.bart.config.pad_token_id

        output_ids = self.tokenizer(outputs)["input_ids"]
        output_ids = [torch.tensor(idx) for idx in output_ids]
        output_ids = pad_sequence(output_ids, batch_first=True, padding_value=self.bart.config.pad_token_id).to(self.device)

        with torch.no_grad():
            # Generate in beam sequences(beam size = batch size)
            output = self.bart.generate(
                input_ids,
                num_beams=batch_size,
                # Diverse beam search
                num_beam_groups=batch_size,
                diversity_penalty=self.diversity_penalty,
                # Output control
                num_return_sequences=batch_size,
                return_dict_in_generate=True,
                output_scores=True,
            )
            sequences = output.sequences.reshape(batch_size, batch_size, -1)[:, :, 1:]
            # Concat output_ids(golden) and beam generated sequences with padding
            output_len, sequences_len = output_ids.size(-1), sequences.size(-1)
            if output_len > sequences_len:
                # output_ids longer
                sequences = torch.cat([sequences, pad_id * torch.ones((sequences.size(0), sequences.size(1), output_len - sequences_len), dtype=torch.long, device=sequences.device)], dim=-1)
            elif output_len < sequences_len:
                # sequences longer
                output_ids = torch.cat([output_ids, pad_id * torch.ones((output_ids.size(0), sequences_len - output_len), dtype=torch.long, device=output_ids.device)], dim=-1)
            assert output_ids.size(-1) == sequences.size(-1)
            sequences = torch.cat([sequences, output_ids.unsqueeze(1)], dim=1).tolist()

            # Sort the outputs in lexical order
            sequences = [sorted(sequence) for sequence in sequences] # sort each beams
            sequences = torch.tensor(sequences, device=input_ids.device) # batch_size * beam_size * seq_len

            # Get boundaries and decoder_mask to obtain the shared prefix
            seq1 = sequences[:,  :-1, :]
            seq2 = sequences[:, 1:, :]
            boundaries = (seq1 != seq2).to(torch.long).argmax(dim=2) # batch_size * batch_size
            decoder_mask = torch.tile(torch.arange(0, sequences.size(2)).unsqueeze(0).unsqueeze(1), (boundaries.size(0), boundaries.size(1), 1)).to(boundaries.device)
            decoder_mask = decoder_mask < boundaries.unsqueeze(2)
            assert torch.equal(seq1 * decoder_mask, seq2 * decoder_mask)
            decoder_prefix = seq1 * decoder_mask
            seq1_idx = torch.gather(seq1, 2, boundaries.unsqueeze(2)) # batch_size * batch_size * 1
            seq2_idx = torch.gather(seq2, 2, boundaries.unsqueeze(2))
            first_diff_tok_idx = torch.cat([seq1_idx, seq2_idx], dim=2) # batch_size * batch_size * 2

            # Rank the outputs
            pibleu_score = get_pibleu_score(input_ids, sequences, self.tokenizer) # batch_size * (batch_size+1)
            rank = torch.argsort(pibleu_score, dim=1)
            # tok_dif_sign: 1 if seq1 score is better, else -1
            tok_dif_sign = 2 * (rank[:, :-1] > rank[:, 1:]) - 1 # batch_size * batch_size

        # Compare adjacent beams
        # we compute single input and its output beams one by one(that's why we set beam_size to batch_size)
        contrast_loss = 0
        cnt = 0
        for i in range(batch_size):
            logits = self.bart(
                input_ids=torch.tile(input_ids[i].unsqueeze(0), (batch_size, 1)),
                attention_mask=attention_mask,
                decoder_input_ids=decoder_prefix[i],
                decoder_attention_mask = decoder_mask[i]
            ).logits # batch_size, seq_len, vocab_size
            logits_gather_index = torch.tile(boundaries[i].unsqueeze(1).unsqueeze(2), (1, 1, logits.size(2)))
            logits = torch.gather(logits, 1, logits_gather_index).squeeze(1) # batch_size, vocab_size
            compare_logits = torch.gather(logits, 1, first_diff_tok_idx[i]) # batch_size, 2
            tok_dif = (compare_logits[:, 0] - compare_logits[:, 1]) * tok_dif_sign[i]
            # loss for input = (0 if tok_dif > contrast_lambda ; else contrast_lambda - tok_dif)
            contrast_loss += torch.sum(self.contrast_lambda - torch.min(torch.ones_like(tok_dif) * self.contrast_lambda, tok_dif))
            cnt += tok_dif.size(0)
        
        assert cnt == batch_size ** 2
        return contrast_loss / cnt

    
    def generate(self, inputs, skip_special_tokens=True):
        batch_size = len(inputs)

        # Tokenize
        input_ids = self.tokenizer(inputs)["input_ids"]
        input_ids = [torch.tensor(idx) for idx in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.bart.config.pad_token_id).to(self.device)

        # Run BART generation
        output = self.bart.generate(
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
