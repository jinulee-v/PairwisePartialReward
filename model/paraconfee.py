import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

class ParaConfee(nn.Module):
    """
    BART based module
    """

    def __init__(self,
            bart: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            num_beams: int = None,
            num_beam_groups: int = None,
            from_batch_neg_examples : int = None,
            diversity_penalty : float = None,
            difference_hinge_lambda : float = None,
            ordering_hinge_lambda : float = None,
            device: torch.device = torch.device("cpu")):
        super(ParaConfee, self).__init__()

        # BART Layer
        self.bart = bart
        self.tokenizer = tokenizer
        # Paraphrase Identification Layer
        # self.pooler = (Simple Average Pooling)
        self.W = nn.Linear(bart.config.hidden_size * 3, 2)

        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups
        self.from_batch_neg_examples = from_batch_neg_examples
        self.diversity_penalty = diversity_penalty
        self.difference_hinge_lambda = difference_hinge_lambda
        self.ordering_hinge_lambda = ordering_hinge_lambda
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

    def get_beam_contrast_loss(self, inputs, outputs, classification_loss_weight=None, distance_loss_weight=None, ordering_loss_weight=None):
        """
        Calculates beam contrastive classification loss(paraphrase detection) / Distance loss(cosine sim marginal loss).
        @param inputs List[str]
        @param outputs List[str]

        @return loss
        """
        batch_size = len(inputs)
        if self.from_batch_neg_examples > batch_size - 1:
            self.from_batch_neg_examples = batch_size - 1

        # Tokenize
        input_ids = self.tokenizer(inputs)["input_ids"]
        input_ids = [torch.tensor(idx) for idx in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.bart.config.pad_token_id).to(self.device)
        attention_mask = input_ids != self.bart.config.pad_token_id

        # Run BART generation
        with torch.no_grad():
            output = self.bart.generate(
                input_ids,
                num_beams=self.num_beams,
                # Diverse beam search
                num_beam_groups=self.num_beam_groups,
                diversity_penalty=self.diversity_penalty,
                # Output control
                num_return_sequences=self.num_beam_groups,
                return_dict_in_generate=True,
                output_scores=True,
            )
            sequences = output.sequences.reshape(batch_size, self.num_beam_groups, -1)

        # Calculate encoder hidden states of the original input
        original_hidden_states = self.bart.model.encoder(
            input_ids,
            attention_mask,
            return_dict=True
        ).last_hidden_state
        # normalize within only attention masks
        original_hidden_states = torch.sum(original_hidden_states * attention_mask.unsqueeze(2), dim=1) / torch.sum(attention_mask, dim=1).unsqueeze(1)

        output_ids = self.tokenizer(outputs)["input_ids"]
        output_ids = [torch.tensor(idx) for idx in output_ids]
        output_ids = pad_sequence(output_ids, batch_first=True, padding_value=self.bart.config.pad_token_id).to(self.device)
        output_attention_mask = output_ids != self.bart.config.pad_token_id
        output_hidden_states = self.bart.model.encoder(
            output_ids,
            output_attention_mask,
            return_dict=True
        ).last_hidden_state
        # normalize within only attention masks
        output_hidden_states = torch.sum(output_hidden_states * output_attention_mask.unsqueeze(2), dim=1) / torch.sum(output_attention_mask, dim=1).unsqueeze(1)

        loss = 0
        for i in range(batch_size):
            # Sample from batch inputs
            batch_random_idx = random.sample(range(batch_size), self.from_batch_neg_examples + 1)
            if i in batch_random_idx:
                batch_random_idx.remove(i)
            else:
                batch_random_idx = batch_random_idx[:-1]
            batch_random_idx = torch.tensor(batch_random_idx).to(self.device)
            batch_hidden_states = original_hidden_states[batch_random_idx, :]

            # Calculate encoder hidden states for beams
            beam_sequences = sequences[i]
            beam_attention_mask = beam_sequences != self.bart.config.pad_token_id
            beam_hidden_states = self.bart.model.encoder(
                beam_sequences,
                beam_attention_mask,
                return_dict=True
            ).last_hidden_state
            beam_hidden_states = torch.sum(beam_hidden_states * beam_attention_mask.unsqueeze(2), dim=1) / torch.sum(beam_attention_mask, dim=1).unsqueeze(1)

            # Concat neg examples(from batch) and pos examples(from beam)
            contrast_hidden_states = torch.cat([batch_hidden_states, beam_hidden_states, output_hidden_states[i].unsqueeze(0)], dim=0)
            contrast_labels = torch.tensor([0] * batch_hidden_states.size(0) + [1] * (beam_hidden_states.size(0) + 1)).to(self.device)

            original_hidden_state = original_hidden_states[i].unsqueeze(0)
            original_hidden_state = torch.tile(original_hidden_state, [contrast_hidden_states.size(0), 1])

            assert contrast_hidden_states.size() == original_hidden_state.size()

            if classification_loss_weight:
                para_detect_input = torch.cat([original_hidden_state, contrast_hidden_states, torch.abs(original_hidden_state-contrast_hidden_states)], dim=1)
                contrast_output = self.W(para_detect_input)
                loss_fn = nn.CrossEntropyLoss()
                loss += loss_fn(contrast_output, contrast_labels) * classification_loss_weight

            if distance_loss_weight:
                cossim = nn.CosineSimilarity(dim=1)(contrast_hidden_states, original_hidden_state)
                min_beam = torch.min(cossim[batch_hidden_states.size(0) : batch_hidden_states.size(0)+self.num_beam_groups])
                max_batch = torch.max(cossim[:batch_hidden_states.size(0)])
                distance_hinge_loss = torch.maximum(torch.tensor([0]).to(self.device), self.difference_hinge_lambda - (min_beam - max_batch))
                loss += distance_hinge_loss.squeeze(0) * distance_loss_weight

            if ordering_loss_weight:
                # Loss function from "CoNT: Contrastive Neural Text Generation"
                cossim = nn.CosineSimilarity(dim=1)(contrast_hidden_states, original_hidden_state)
                sorted_cossim, sorted_cossim_orders = torch.sort(cossim, descending=True)
                order_loss = 0
                # TODO parallelize this code
                for i in range(sorted_cossim_orders.size(0) - 1):
                    for j in range(i, sorted_cossim_orders.size(0)):
                        order_diff = sorted_cossim_orders[i] - sorted_cossim_orders[j]; order_diff = order_diff if order_diff > 0 else -order_diff
                        value_diff = torch.abs(sorted_cossim[i] - sorted_cossim[j])
                        order_loss += torch.max(torch.tensor([0]).to(self.device), order_diff * self.ordering_hinge_lambda - value_diff)
                loss += order_loss.squeeze(0) * self.ordering_hinge_lambda

        return loss / batch_size
    
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
            # Diverse Beam Search
            num_beam_groups=self.num_beam_groups,
            diversity_penalty=self.diversity_penalty,
            # Output control
            max_new_tokens=int(input_ids.size(1) * 1.5),
            num_return_sequences=self.num_beam_groups,
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
            for __ in range(self.num_beam_groups):
                results[-1].append(output[i])
                i += 1
        return results

    def classify(self, inputs1, inputs2):
        assert len(inputs1) == len(inputs2)

        # Input1
        # Tokenize
        input_ids1 = self.tokenizer(inputs1)["input_ids"]
        input_ids1 = [torch.tensor(idx) for idx in input_ids1]
        input_ids1 = pad_sequence(input_ids1, batch_first=True, padding_value=self.bart.config.pad_token_id).to(self.device)
        attention_mask1 = input_ids1 != self.bart.config.pad_token_id

        hidden_states1 = self.bart.model.encoder(
            input_ids1,
            attention_mask1,
            return_dict=True
        ).last_hidden_state
        # normalize within only attention masks
        hidden_states1 = torch.sum(hidden_states1 * attention_mask1.unsqueeze(2), dim=1) / torch.sum(attention_mask1, dim=1).unsqueeze(1)

        # Input2
        # Tokenize
        input_ids2 = self.tokenizer(inputs2)["input_ids"]
        input_ids2 = [torch.tensor(idx) for idx in input_ids2]
        input_ids2 = pad_sequence(input_ids2, batch_first=True, padding_value=self.bart.config.pad_token_id).to(self.device)
        attention_mask2 = input_ids2 != self.bart.config.pad_token_id

        hidden_states2 = self.bart.model.encoder(
            input_ids2,
            attention_mask2,
            return_dict=True
        ).last_hidden_state
        # normalize within only attention masks
        hidden_states2 = torch.sum(hidden_states2 * attention_mask2.unsqueeze(2), dim=1) / torch.sum(attention_mask2, dim=1).unsqueeze(1)

        # Run siamese classification layer
        para_detect_input = torch.cat([hidden_states1, hidden_states2, torch.abs(hidden_states1-hidden_states2)], dim=1)
        results = self.W(para_detect_input)

        return results