import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

class ParaphraserBase(nn.Module):
    """
    BART based module
    """

    def __init__(self,
            base: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            num_beams: int = None,
            device: torch.device = torch.device("cpu"), **kwargs):
        super(ParaphraserBase, self).__init__()

        # BART Layer
        self.base = base
        self.tokenizer = tokenizer
        self.pad_id = self.base.config.pad_token_id

        self.num_beams = num_beams
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

    def synonym_branching_test(self, inputs, output_prefixes, original, synonym):
        """
        Script for testing synonym branching.
        @param inputs List[str]
        @param output_prefixes List[str]
        @param original List[str]
        @param synonym List[str]

        @return loss
        """
        # Transform batch to list
        inputs, output_prefixes, original, synonym = list(inputs), list(output_prefixes), list(original), list(synonym)

        # Tokenize
        input_ids = self.tokenizer(inputs, truncation=True)["input_ids"]
        input_ids = [torch.tensor(idx, device=self.device) for idx in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        input_attention_mask = input_ids != self.pad_id

        output_ids = self.tokenizer(output_prefixes, truncation=True)["input_ids"]
        output_ids = [torch.tensor(idx, device=self.device) for idx in output_ids]
        output_ids = pad_sequence(output_ids, batch_first=True, padding_value=self.pad_id)
        output_attention_mask = output_ids != self.pad_id
        boundaries = torch.sum(output_attention_mask, dim=1)-1

        original_ids = self.tokenizer(original)["input_ids"]
        synonym_ids = self.tokenizer(synonym)["input_ids"]
        first_diff_tok_idx = []
        for o, s in zip(original_ids, synonym_ids):
            i = 1
            try:
                while o[i] == s[i]:
                    i+=1
                assert o[i] != s[i]
                first_diff_tok_idx.append([o[i], s[i]])
            except IndexError:
                raise ValueError(f"original & synonym must be different: original={self.tokenizer.decode(o)}, synonym={self.tokenizer.decode(s)}")
        first_diff_tok_idx = torch.tensor(first_diff_tok_idx, dtype=torch.long, device=self.device)

        logits = self.base(
            input_ids=input_ids,
            attention_mask=input_attention_mask,
            decoder_input_ids=output_ids,
            decoder_attention_mask = output_attention_mask
        ).logits # batch_size, seq_len, vocab_size
        logits_gather_index = torch.tile(boundaries.unsqueeze(1).unsqueeze(2), (1, 1, logits.size(2)))
        logits = torch.gather(logits, 1, logits_gather_index).squeeze(1) # batch_size, vocab_size
        logits = F.log_softmax(logits, dim=1)
        compare_logits = torch.gather(logits, 1, first_diff_tok_idx) # batch_size, 2
        tok_dif = compare_logits[:, 1] - compare_logits[:, 0]
        # tok_diff = logp(synonym) - logp(original)
        
        return tok_dif