import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    LogitsProcessorList
)
from .dataset import get_prefix

class ParaphraserBase(nn.Module):
    """
    BART based module
    """

    def __init__(self,
            base: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            num_beams: int = None,
            # device: torch.device = torch.device("cpu"),
            **kwargs):
        super(ParaphraserBase, self).__init__()

        # BART Layer
        self.base = base
        self.tokenizer = tokenizer
        self.pad_id = self.base.config.pad_token_id

        self.num_beams = num_beams
        # self.device = device

    def get_generation_loss(self, inputs, outputs):
        """
        Calculates classic teacher-forced generation loss.
        @param inputs List[str]
        @param outputs List[str]

        @return loss
        """
        # torch.cuda.empty_cache()
        # assert len(inputs) == len(outputs)
        # batch_size = len(inputs)

        # Tokenize
        # inputs = {k:v.to(self.device) for k,v in self.tokenizer(inputs, return_tensors='pt', padding=True).items()}
        # input_ids = [torch.tensor(idx) for idx in input_ids]
        # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id).to(self.device)
        # attention_mask = input_ids != self.pad_id
        # decoder_input_ids = self.tokenizer(outputs, truncation=True)["input_ids"]
        # decoder_input_ids = [torch.tensor(idx) for idx in decoder_input_ids]
        # decoder_input_ids = pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.pad_id).to(self.device)
        # decoder_attention_mask = decoder_input_ids != self.pad_id
        # target = self.tokenizer(outputs, return_tensors='pt', padding=True)['input_ids'].to(self.device)

        # Run forward pass with teacher forcing
        loss = self.base(
            inputs.to(self.base.device),
            labels=outputs.to(self.base.device),
        ).loss
        return loss
    
    def generate(self, inputs, skip_special_tokens=True, sampling=False, **kwargs):
        batch_size = len(inputs)

        # Tokenize
        # input_ids = self.tokenizer(inputs, truncation=True)["input_ids"]
        # input_ids = [torch.tensor(idx) for idx in input_ids]
        # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id).to(self.device)

        # Run BART generation: DFA based generation
        if sampling:
            new_inputs = inputs.to(self.base.device).repeat(self.num_beams, 1)
            output = self.base.generate(
                new_inputs,
                do_sample=True,
                num_beams=1,
                # Output control
                max_new_tokens=int(inputs.shape[1] * 1.5),
                num_return_sequences=self.num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs,
            )
        else:
            output = self.base.generate(
                inputs.to(self.base.device),
                num_beams=self.num_beams,
                # Output control
                max_new_tokens=int(inputs.shape[1] * 1.5),
                num_return_sequences=self.num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs,
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

    def generate_ngram_constrained(self, inputs, logits_processor, skip_special_tokens=True):
        """
        Implements various constrained decoding that differentiates the output from input.
        Applies penalty to certain repetitions from input.
        """
        batch_size = len(inputs)

        # Tokenize
        input_ids = self.tokenizer(inputs, truncation=True)["input_ids"]
        input_ids = [torch.tensor(idx) for idx in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id).to(self.base.device)

        # Set LogitsProcessor
        logits_processor.exclude_id=[self.pad_id, self.tokenizer.eos_token_id]
        logits_processor.update(input_ids)
        logits_processors = LogitsProcessorList([logits_processor])

        # Run BART generation
        output = self.base.generate(
            input_ids,
            num_beams=self.num_beams,
            # Output control
            max_new_tokens=int(input_ids.size(1) * 1.5),
            num_return_sequences=self.num_beams,
            return_dict_in_generate=True,
            output_scores=True,
            # N-gram penalize
            logits_processor=logits_processors
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

    def branching_test(self, src, metric=None):
        """
        Calculates the token_wise contrastive loss.
        @return loss
        """
        if hasattr(self, 'metric'):
            metric = self.metric

        batch_size, _ = src.shape
        # beam_size = hypos[0].shape[0]
        beam_size = self.num_beams

        with torch.no_grad():
            # Generate in beam sequences(beam size = batch size)
            output = self.base.generate(
                src.to(self.base.device),
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
                sequences = sequences[:, :, bos_index:].contiguous()
            else:
                pad_index = sequences[0, 0].tolist().index(self.tokenizer.pad_token_id)
                sequences = sequences[:, :, pad_index+1:].contiguous()

            samples_str = self.tokenizer.batch_decode(sequences.view(-1, sequences.size(-1)), skip_special_tokens=True) # aggregate batch & sample IDs
            hypos = sequences
            sequences = sequences.tolist()

        # Rank the outputs
        sources_decode = self.tokenizer.batch_decode(src, skip_special_tokens=True) # [B]
        extended_inputs = [x for x in sources_decode for _ in range(beam_size)]

        scores = metric(extended_inputs, None, samples_str, (batch_size, beam_size), extended=True).reshape(batch_size, beam_size).cpu() # batch_size * num_beams
        # Extract common prefixes out of the prefix tree
        all_branches, all_win_indices, all_lose_indices = get_prefix(sequences, scores, self.tokenizer.pad_token_id)

        win_count = [0 for _ in range(hypos.size(2))]; total_count = [0 for _ in range(hypos.size(2))]
        # cnt = 0
        for i in range(batch_size):
            ith_input_ids = src[i].repeat(beam_size, 1)
            branches = all_branches[i]
            win_indices = all_win_indices[i]
            lose_indices = all_lose_indices[i]
            target = hypos[i]
            logits = self.base(
                input_ids=ith_input_ids.to(self.base.device),
                labels=target.to(self.base.device),
            ).logits # num_beams, seq_len, vocab_size
            probs = logits.softmax(dim=-1).reshape(-1, logits.shape[-1]) # [B*T,V]
            probs = torch.gather(probs, -1, index=target.reshape(-1).unsqueeze(-1)).reshape(target.shape) # [B, T]
            lose_x, lose_y = zip(*lose_indices)
            win_x, win_y = zip(*win_indices)
            wins = (probs[lose_x, lose_y] < probs[win_x, win_y]).to(torch.long)
            win_count[branches[i][0]] += torch.sum(wins)
            total_count[branches[i][0]] += torch.numel(wins)
        
        return win_count, total_count
    
    def token_rank(self, src, metric=None):
        if hasattr(self, 'metric'):
            metric = self.metric

        batch_size, _ = src.shape
        # beam_size = hypos[0].shape[0]
        beam_size = self.num_beams

        with torch.no_grad():
            # Generate in beam sequences(beam size = batch size)
            output = self.base.generate(
                src.to(self.base.device),
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
                sequences = sequences[:, :, bos_index:].contiguous()
            else:
                pad_index = sequences[0, 0].tolist().index(self.tokenizer.pad_token_id)
                sequences = sequences[:, :, pad_index+1:].contiguous()

            samples_str = self.tokenizer.batch_decode(sequences.view(-1, sequences.size(-1)), skip_special_tokens=True) # aggregate batch & sample IDs

            token_scores = torch.stack(output.scores, dim=1)
            token_scores = token_scores.reshape(batch_size, beam_size, -1, token_scores.size(2))
            # batch_size, beam_size, seq_len, vocab_size

            if token_scores.size(2) != sequences.size(2):
                print(token_scores.size(2), sequences.size(2))
                exit()
            

        # Rank the outputs with oracle -> Deprecated
        # sources_decode = self.tokenizer.batch_decode(src, skip_special_tokens=True) # [B]
        # extended_inputs = [x for x in sources_decode for _ in range(beam_size)] 
        # scores = metric(extended_inputs, None, samples_str, (batch_size, beam_size), extended=True).reshape(batch_size, beam_size) # batch_size * num_beams
        # best_seq_idx = torch.argmax(scores, dim=1)
        # Instead, select the best token
        best_seq_idx = torch.zeros((batch_size), dtype=torch.long, device=sequences.device)
        # best_seq_idx: batch_size
        best_seq = torch.gather(sequences, 1, best_seq_idx.unsqueeze(1).unsqueeze(2).tile(1, 1, sequences.size(-1))).squeeze(1)
        # best_seq: batch_size, max_len

        # batch_size, beam_size, seq_len, vocab_size
        best_seq_logit_rank = torch.gather(token_scores, 1, best_seq_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).tile(1, 1, token_scores.size(2), self.base.config.vocab_size)).squeeze(1)
        # batch_size, seq_len, vocab_size
        def get_rank(x, indices):
            vals = x[range(x.size(0)), indices]
            return (x > vals[:, None]).long().sum(dim=1)
        best_seq_logit_rank = get_rank(best_seq_logit_rank.reshape(-1, self.base.config.vocab_size), best_seq.reshape(-1))
        best_seq_logit_rank = best_seq_logit_rank.reshape(best_seq.size(0), best_seq.size(1))
        best_seq_logit_rank = (1 / (1+best_seq_logit_rank)) # Inverse rank
        
        # Length filtering
        LENGTH=10
        best_seq_logit_rank *= (best_seq[:, LENGTH] == self.tokenizer.eos_token_id).unsqueeze(1)
        best_seq *= (best_seq[:, LENGTH] == self.tokenizer.eos_token_id).unsqueeze(1)
        
        best_seq_logit_rank *= ((best_seq) != 0).to(torch.long)

        # return sum of average, element that is zero(greedy top rank), sum
        return torch.sum(best_seq_logit_rank, dim=0), torch.sum(best_seq != 0, dim=0) # batch-sum, length info preserved
        