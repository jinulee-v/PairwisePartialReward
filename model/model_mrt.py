import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

from .model import ParaphraserBase
torch.autograd.set_detect_anomaly(True)
class Paraphraser(ParaphraserBase):
    """
    Implementation of MRT(Minimum Risk Training) for diverse paraphrase generation
    """

    def __init__(self,
            base: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            metric: callable,
            num_beams: int = None,
            sample_size: int = None,
            device: torch.device = torch.device("cpu"), **kwargs):
        super(Paraphraser, self).__init__(base, tokenizer, num_beams=num_beams, device=device)

        # BART Layer
        self.base = base
        self.tokenizer = tokenizer
        self.metric = metric
        self.pad_id = self.base.config.pad_token_id

        self.num_beams = num_beams
        self.sample_size = sample_size
        self.device = device


    def get_contrastive_loss(self, inputs, outputs):
        """
        Calculates the 'Minimum Risk Training' loss.
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
        with torch.no_grad():
            # Batchified sequence loss calcuiation
            sequences = None
            for start in range(0, self.sample_size, batch_size):
                end = min(start + batch_size, self.sample_size)
                output = self.base.generate(
                    input_ids,
                    # Output control
                    num_return_sequences=end - start,
                    return_dict_in_generate=True,
                    do_sample=True
                ).sequences.reshape(batch_size, end-start, -1)[:, :, 1:]
                # Append to sequences
                if sequences is None:
                    sequences = output
                else:
                    # pad to longer tensor
                    if sequences.size(2) > output.size(2):
                        padder = torch.ones((output.size(0), output.size(1), sequences.size(2)-output.size(2)), device=output.device, dtype=torch.long) * self.pad_id
                        output = torch.cat((output, padder), dim=2)
                    elif output.size(2) > sequences.size(2):
                        padder = torch.ones((sequences.size(0), sequences.size(1), output.size(2)-sequences.size(2)), device=sequences.device, dtype=torch.long) * self.pad_id
                        sequences = torch.cat((sequences, padder), dim=2)
                    # append
                    sequences = torch.cat((sequences, output), dim=1)
        
        # Minimum Risk Training
        mrt_loss = 0
        for i in range(batch_size):
            # Deduplicate sampled sequences
            sequences_dedup = torch.unique(sequences[i], dim=0)
            decoder_mask = sequences_dedup != self.pad_id
            # vanishing-prevention
            # vanish_prevent = sequences_dedup.size(1) * 4
            
            # Get PiBLEU score
            # pibleu_score = 1 - get_pibleu_score(input_ids[i].unsqueeze(0), sequences_dedup.unsqueeze(0), self.tokenizer)[0] # batch_size * num_beams
            # Rank the outputs
            samples_str = self.tokenizer.batch_decode(sequences_dedup, skip_special_tokens=True) # aggregate batch & sample IDs
            samples_str = [[s] for s in samples_str]
            # samples_str = [samples_str[n:n+sequences_dedup.size(1)] for n in range(0, sequences_dedup.size(0), sequences_dedup.size(1))] # Restructure outputs
            metrics = self.metric([inputs[i]] * len(samples_str), [outputs[i]] * len(samples_str), samples_str) # batch_size * num_beams

            log_probs = None
            # Batchified sequence loss calculation
            for start in range(0, sequences_dedup.size(0), batch_size):
                end = min(start + batch_size, sequences_dedup.size(0))
                # print(i, sequences_dedup.size(), start, end)
                logits = self.base(
                    input_ids=torch.tile(input_ids[i].unsqueeze(0), (end-start, 1)),
                    attention_mask=torch.tile(attention_mask[i].unsqueeze(0), (end-start, 1)),
                    decoder_input_ids=sequences_dedup[start:end],
                    decoder_attention_mask=decoder_mask[start:end]
                ).logits # (end-start) * seq_len * vocab_size
                
                # Calculate NLL losses(=log probability)
                log_prob = - loss_fct(logits.reshape(-1, logits.size(2)), sequences_dedup[start:end].reshape(-1))
                log_prob = log_prob.reshape(logits.size(0), logits.size(1)) * decoder_mask[start:end] # num_beams * seq_len
                log_prob = torch.sum(log_prob, dim=1) # num_beams

                if log_probs is None:
                    log_probs = log_prob
                else:
                    log_probs = torch.cat((log_probs, log_prob), dim=0)
                
            log_probs -= torch.mean(log_probs) # normalize to prevent underflow NaN
            probs = torch.exp(log_probs) # num_beams
            
            # Calculate loss
            # print(prob, pibleu_score[start:end])
            total_loss_per_sample = torch.sum(probs * metrics)
            # Accumulate sample probabilities
            total_sample_prob = torch.sum(probs)
            
            if torch.isfinite(total_loss_per_sample) and torch.isfinite(total_sample_prob): # NaN prevention
                mrt_loss += total_loss_per_sample / total_sample_prob

        return mrt_loss / batch_size