import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

from .model import ParaphraserBase

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
        # best_token = None
        # not_best_tokens = []
        # for token, value in subtree.items():
        #     if best_token is None:
        #         best_token = (token, value[0])
        #     else:
        #         if rank[value[0]] > rank[best_token[1]]:
        #             not_best_tokens.append(best_token[0])
        #             best_token = (token, value[0])
        #         else:
        #             not_best_tokens.append(token)
        # for not_best_token in not_best_tokens:
        #     results.append((curr_seq[:], best_token[0], not_best_token))
        tokens = []
        for token, value in subtree.items():
            tokens.append((token, value[0]))
        tokens = sorted(tokens, key=lambda x: rank[x[1]])
        for i in range(len(tokens) - 1):
            results.append((curr_seq[:], tokens[i+1][0], tokens[i][0]))

    for token, value in subtree.items():
        curr_seq.append(token)
        _dfs(value[1], rank, curr_seq, results)
        curr_seq.pop()

class Paraphraser(ParaphraserBase):
    """
    Implementation of TrieCL(proposed) for diverse paraphrase generation
    """

    def __init__(self,
            base: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            metric: callable,
            num_beams: int = None,
            contrast_lambda : float = None,
            device: torch.device = torch.device("cpu"), **kwargs):
        super(Paraphraser, self).__init__(base, tokenizer, num_beams=num_beams, device=device)

        # BART Layer
        self.base = base
        self.tokenizer = tokenizer
        self.metric = metric
        self.pad_id = self.base.config.pad_token_id
        self.eos_id = self.base.config.eos_token_id
        self.bos_id = self.base.config.bos_token_id
        if self.bos_id is None:
            self.bos_id = self.pad_id # T5 hotfix

        self.num_beams = num_beams
        self.contrast_lambda = contrast_lambda
        self.device = device


    def get_prefix(self, sequences, ranks):
        prefixes = []
        first_diff_tok_idx = []
        for batch, rank in zip(sequences, ranks):
            # Build trie
            trie = {}
            for seq_id, seq in enumerate(batch):
                curr_trie = trie
                not_first_tok = False
                for tok in seq:
                    if tok not in curr_trie:
                        curr_trie[tok] = [seq_id, {}]
                    # Keep track of beam ID with highest score
                    curr_trie[tok][0] = seq_id if rank[seq_id] > rank[curr_trie[tok][0]] else curr_trie[tok][0]
                    curr_trie = curr_trie[tok][1] 
                    if not_first_tok and tok in [self.pad_id]:
                        break
                    not_first_tok = True
            # Extract prefix pairs and the branching token
            prefix_token_pairs = []
            _dfs(trie, rank, [], prefix_token_pairs)

            beam_size = len(rank)
            while len(prefix_token_pairs) < beam_size:
                # Patch for (rare) cases prefix_token_pair size is not consistent
                prefix_token_pairs.append(([self.pad_id], self.pad_id, self.pad_id))
            assert len(prefix_token_pairs) == beam_size

            prefixes.append([torch.tensor(pair[0], dtype=torch.long) for pair in prefix_token_pairs])
            first_diff_tok_idx.append(torch.tensor([[pair[1], pair[2]] for pair in prefix_token_pairs]).unsqueeze(0))

        prefixes = [pad_sequence(prefix, batch_first=True, padding_value=self.tokenizer.pad_token_id).transpose(0, 1) for prefix in prefixes]
        prefixes = pad_sequence(prefixes, batch_first=True, padding_value=self.tokenizer.pad_token_id).transpose(1, 2)
        first_diff_tok_idx = torch.cat(first_diff_tok_idx, dim=0)

        # return prefixes, first_diff_tok_idx
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
            if self.tokenizer.bos_token_id is not None:
                bos_index = sequences[0, 0].tolist().index(self.tokenizer.bos_token_id)
                sequences = sequences[:, :, bos_index:]

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

            # Extract common prefixes out of the prefix tree
            decoder_prefix, first_diff_tok_idx = self.get_prefix(sequences.tolist(), ranks)
            decoder_prefix = decoder_prefix.to(self.device)
            first_diff_tok_idx = first_diff_tok_idx.to(self.device)

            # Get boundaries and decoder_mask to obtain the shared prefix
            decoder_mask = (decoder_prefix != self.tokenizer.pad_token_id).long()
            boundaries = torch.sum(decoder_mask[:, :, 1:], dim=-1)

        # Compare adjacent beams
        # we compute single input and its output beams one by one(that's why we set beam_size to batch_size)
        contrast_loss = 0
        cnt = 0
        for i in range(batch_size):
            ith_input_ids = torch.tile(input_ids_list[i].unsqueeze(0), (self.num_beams, 1))
            logits = self.base(
                input_ids=ith_input_ids,
                attention_mask=(ith_input_ids != self.pad_id),
                decoder_input_ids=decoder_prefix[i, :, :],
                decoder_attention_mask = decoder_mask[i, :, :]
            ).logits # num_beams, seq_len, vocab_size
            logits_gather_index = torch.tile(boundaries[i].unsqueeze(1).unsqueeze(2), (1, 1, logits.size(2)))
            logits = torch.gather(logits, 1, logits_gather_index).squeeze(1) # num_beams, vocab_size
            compare_logits = torch.gather(logits, 1, first_diff_tok_idx[i]) # num_beams, 2
            tok_dif = compare_logits[:, 0] - compare_logits[:, 1]
            # loss for input = (0 if tok_dif > contrast_lambda ; else contrast_lambda - tok_dif)
            update_val = torch.sum(self.contrast_lambda - torch.min(torch.ones_like(tok_dif) * self.contrast_lambda, tok_dif))
            if not torch.isnan(update_val): # NaN prevention
                contrast_loss += update_val
            cnt += tok_dif.size(0)
        
        assert cnt == batch_size * self.num_beams
        return contrast_loss / cnt
