import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

from .pibleu import get_pibleu_score

def _dfs(subtree, rank, curr_seq, results):
    """
    DFS function for Trie traversal.
    """
    # Reached an end
    if len(subtree) == 0:
        return

    # Branching trie
    if len(subtree) > 1:
        # Find the branch with highest rank
        best_token = None
        not_best_tokens = []
        for token, value in subtree.items():
            if best_token is None:
                best_token = (token, value[0])
            else:
                if rank[value[0]] > best_token[1]:
                    not_best_tokens.append(best_token[0])
                    best_token = (token, value[0])
                else:
                    not_best_tokens.append(token)
        for not_best_token in not_best_tokens:
            results.append((curr_seq[:], best_token[0], not_best_token))

    for token, value in subtree.items():
        curr_seq.append(token)
        _dfs(value[1], rank, curr_seq, results)
        curr_seq.pop()

class Paraphraser(nn.Module):
    """
    BART based module
    """

    def __init__(self,
            base: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            num_beams: int = None,
            contrast_lambda : float = None,
            device: torch.device = torch.device("cpu")):
        super(Paraphraser, self).__init__()

        # BART Layer
        self.base = base
        self.tokenizer = tokenizer
        self.pad_id = self.base.config.pad_token_id

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

    def get_prefix(self, sequences, ranks):

        prefixes = []
        first_diff_tok_idx = []
        for batch, rank in zip(sequences, ranks):
            # Build trie
            trie = {}
            for seq_id, seq in enumerate(batch):
                curr_trie = trie
                for tok in seq:
                    if tok not in curr_trie:
                        curr_trie[tok] = [seq_id, {}]
                    # Keep track of beam ID with highest score
                    curr_trie[tok][0] = seq_id if rank[seq_id] > rank[curr_trie[tok][0]] else curr_trie[tok][0]
                    curr_trie = curr_trie[tok][1] 
                    if tok == self.tokenizer.pad_token_id:
                        break
            # Extract prefix pairs and the branching token
            prefix_token_pairs = []
            _dfs(trie, rank, [], prefix_token_pairs)
            
            beam_size = len(rank) - 1
            while len(prefix_token_pairs) < beam_size:
                # Patch for (rare) cases prefix_token_pair size is not consistent
                prefix_token_pairs.append(([self.tokenizer.bos_token_id, self.tokenizer.eos_token_id], 3, 3))
            assert len(prefix_token_pairs) == beam_size

            prefixes.append([torch.tensor(pair[0]) for pair in prefix_token_pairs])
            first_diff_tok_idx.append(torch.tensor([[pair[1], pair[2]] for pair in prefix_token_pairs]).unsqueeze(0))

        prefixes = [pad_sequence(prefix, batch_first=True, padding_value=self.tokenizer.pad_token_id).transpose(0, 1) for prefix in prefixes]
        prefixes = pad_sequence(prefixes, batch_first=True, padding_value=self.tokenizer.pad_token_id).transpose(1, 2)
        first_diff_tok_idx = torch.cat(first_diff_tok_idx, dim=0)

        return prefixes, first_diff_tok_idx

    def get_contrastive_loss(self, inputs, outputs):
        """
        Calculates the token_wise contrastive loss.
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

        with torch.no_grad():
            # Generate in beam sequences(beam size = batch size)
            output = self.base.generate(
                input_ids,
                num_beams=batch_size + 1,
                # Output control
                num_return_sequences=batch_size + 1,
                return_dict_in_generate=True,
                output_scores=True,
            )
            sequences = output.sequences.reshape(batch_size, batch_size + 1, -1)[:, :, 1:]

            # Rank the outputs
            pibleu_score = get_pibleu_score(input_ids, sequences, self.tokenizer) # batch_size * (batch_size+1)
            ranks = torch.argsort(pibleu_score, dim=1).to(torch.device('cpu'), non_blocking=True).tolist()

            # Extract common prefixes out of the prefix tree
            decoder_prefix, first_diff_tok_idx = self.get_prefix(sequences.to(torch.device('cpu'), non_blocking=True).tolist(), ranks)
            decoder_prefix = decoder_prefix.to(self.device, non_blocking=True)
            first_diff_tok_idx = first_diff_tok_idx.to(self.device, non_blocking=True)
                
            # Get boundaries and decoder_mask to obtain the shared prefix
            decoder_mask = (decoder_prefix != self.tokenizer.pad_token_id).long()
            boundaries = torch.sum(decoder_mask, dim=-1) - 1

        # Compare adjacent beams
        # we compute single input and its output beams one by one(that's why we set beam_size to batch_size)
        contrast_loss = 0
        cnt = 0
        for i in range(batch_size):
            logits = self.base(
                input_ids=torch.tile(input_ids[i].unsqueeze(0), (batch_size, 1)),
                attention_mask=torch.tile(attention_mask[i].unsqueeze(0), (batch_size, 1)),
                decoder_input_ids=decoder_prefix[i],
                decoder_attention_mask = decoder_mask[i]
            ).logits # batch_size, seq_len, vocab_size
            logits_gather_index = torch.tile(boundaries[i].unsqueeze(1).unsqueeze(2), (1, 1, logits.size(2)))
            logits = torch.gather(logits, 1, logits_gather_index).squeeze(1) # batch_size, vocab_size
            compare_logits = torch.gather(logits, 1, first_diff_tok_idx[i]) # batch_size, 2
            tok_dif = compare_logits[:, 0] - compare_logits[:, 1]
            # loss for input = (0 if tok_dif > contrast_lambda ; else contrast_lambda - tok_dif)
            contrast_loss += torch.sum(self.contrast_lambda - torch.min(torch.ones_like(tok_dif) * self.contrast_lambda, tok_dif))
            cnt += tok_dif.size(0)
        
        assert cnt == batch_size ** 2
        return contrast_loss / cnt
    
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
