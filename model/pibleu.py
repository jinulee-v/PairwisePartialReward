"""
PiBLEU scoring evaluation
"""
import json
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import sentence_bleu

batch_size=32

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
hg_model_hub_name = "textattack/bert-base-uncased-QQP"
pi_tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
pi_model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name).to(device)
pad_id = pi_tokenizer.pad_token_id
entail_label_id = 1

def load_jsonl(path):
    inst_list = []
    with open(path) as f:
        for line in f:
            inst_list.append(json.loads(line))
    return inst_list

def get_para_score(s1, s2, device):
    global pi_model, pi_tokenizer
    pi_model = pi_model.to(device)
    assert len(s1) == len(s2)
    length = len(s1)
    tokenized_input_seq_pair = pi_tokenizer.batch_encode_plus(list(zip(s1, s2)), max_length=256, return_token_type_ids=True, truncation=True)
    input_ids = [torch.tensor(x) for x in tokenized_input_seq_pair["input_ids"]]
    token_type_ids = [torch.tensor(x) for x in tokenized_input_seq_pair["token_type_ids"]]
    attention_masks = [torch.tensor(x) for x in tokenized_input_seq_pair["attention_mask"]]
    predicted_probability = None
    for head in range(0, length - 1, batch_size):
        tail = min(head + batch_size, length)
        with torch.no_grad():
            input_id = pad_sequence(input_ids[head:tail], batch_first=True, padding_value=pad_id).to(device)
            token_type_id = pad_sequence(token_type_ids[head:tail], batch_first=True, padding_value=0).to(device)
            attention_mask = pad_sequence(attention_masks[head:tail], batch_first=True, padding_value=0).to(device)
            outputs = pi_model(
                input_id,
                attention_mask=attention_mask,
                token_type_ids=token_type_id,
                labels=None,
            )
            if predicted_probability is None:
                predicted_probability = torch.softmax(outputs[0], dim=1)
            else:
                predicted_probability = torch.cat((predicted_probability, torch.softmax(outputs[0], dim=1)), dim=0)
    return predicted_probability

def get_bleu_score(s1, s2):
    scores = [sentence_bleu([x1.split()], x2.split(), weights=(0.25,0.25,0.25,0.25)) for x1, x2 in zip(s1, s2)]
    return torch.tensor(scores)

def form_ngram(input_tensor, n=2):
    """
    input_tensor: batch x sample_num x seq_len
    return: batch x seq_len-3 x 4
    """
    bsz, cand_num, seq_len = input_tensor.size(0), input_tensor.size(1), input_tensor.size(2)
    seq_len_clip = seq_len - n + 1
    input_tensor_repeated = input_tensor[:, :, None, :].repeat(1, 1, seq_len_clip, 1)
    help_matrix_1 = torch.triu(torch.ones(seq_len, seq_len))
    help_matrix_2 = torch.triu(torch.ones(seq_len, seq_len), diagonal=n)
    help_matrix = (help_matrix_1 - help_matrix_2)[:seq_len_clip].bool()[None, None, :, :]
    ret_tensor = torch.masked_select(input_tensor_repeated, help_matrix.to(input_tensor.device))
    return ret_tensor.view(bsz, cand_num, seq_len_clip, n)

def _torch_bleu(ref_tensor, sys_tensor, pad_id, n_gram=2):
    """
    ref_tensor: batch x seq_len1
    sys_tensor: batch x sample_num x seq_len2
    """
    sys_padding = (~(sys_tensor == pad_id)).float()
    ref_padding = (~(ref_tensor == pad_id)).float()
    n = min(min(n_gram, ref_tensor.size(-1)), sys_tensor.size(-1))
    ref_lengths = torch.sum(ref_padding, dim=-1) - n + 1
    ref_ones = torch.ones_like(ref_lengths, device=ref_lengths.device)
    ref_lengths = torch.where(ref_lengths > 0, ref_lengths, ref_ones)
    sys_lengths = torch.sum(sys_padding, dim=-1) - n + 1
    sys_ones = torch.ones_like(sys_lengths, device=sys_lengths.device)
    sys_lengths = torch.where(sys_lengths > 0, sys_lengths, sys_ones)
    ref_tensor = ref_tensor * ref_padding
    bsz, sample_num = sys_tensor.size(0), sys_tensor.size(1)
    ref_tensor = ref_tensor[:, None, :].repeat(1, sample_num, 1)
    input_tensor1_4gram = form_ngram(ref_tensor, n).float()
    input_tensor2_4gram = form_ngram(sys_tensor, n).float()  # batch x sample_num x seq_len-3 x 4
    sim_matrix = torch.cosine_similarity(input_tensor2_4gram.unsqueeze(3), input_tensor1_4gram.unsqueeze(2),
                                            dim=-1) >= 1.0
    sim_matrix = torch.sum(torch.max(sim_matrix, dim=-1).values, dim=-1)
    length = sys_lengths + ref_lengths.unsqueeze(1)
    return sim_matrix / length  # batch x sample_num

def torch_bleu(ref_tensor, sys_tensor, pad_id, n_gram=2):
    # If short enough, do not batchify
    if ref_tensor.size(0) < 2 * batch_size:
        return _torch_bleu(ref_tensor, sys_tensor, pad_id, n_gram)

    # Divide by batch
    bleu = None
    for head in range(0, ref_tensor.size(0), batch_size):
        tail = min(head + batch_size, ref_tensor.size(0))

        ref = ref_tensor[head:tail]
        sys = sys_tensor[head:tail]
        
        batch_bleu = _torch_bleu(ref, sys, pad_id, n_gram)
        if bleu is None:
            bleu = batch_bleu
        else:
            bleu = torch.cat([bleu, batch_bleu], dim=0)
    return bleu

@torch.no_grad()
def get_pibleu_score(target_inp, samples_all, tokenizer, eval=False):
    if type(target_inp[0]) is str:
        target_inp = tokenizer(target_inp, padding=True, truncation=True, return_tensors='pt')["input_ids"].to(device)
    if type(samples_all[0][0]) is str: # List[List[str]]
        samples_tensors = []
        beam_size = len(samples_all[0])
        for samples in samples_all:
            samples_tensors.extend(tokenizer(samples, padding=True, truncation=True, return_tensors='pt')["input_ids"])
        samples_all = pad_sequence(samples_tensors, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        samples_all = samples_all.reshape(-1, beam_size, samples_all.size(-1))
    
    # target_inp: batch_size * seq_len1
    # samples_all: batch_size * beam size * seq_len2

    batch_size = samples_all.size(0) # batch_size
    sample_size = samples_all.size(1) # sample # per each batch elements

    # Bleu score
    bleu_score = torch_bleu(target_inp, samples_all, pad_id=tokenizer.pad_token_id)

    # Paraphrase identification
    target_str = tokenizer.batch_decode(target_inp, skip_special_tokens=True)
    target_str = [item for item in target_str for _ in range(sample_size)]  # repeat each target_inp sentences N times
    samples_str = tokenizer.batch_decode(samples_all.view(-1, samples_all.size(-1)), skip_special_tokens=True) # aggregate batch & sample IDs
    assert len(target_inp) == len(samples_all)
    para_score = torch.argmax(get_para_score(samples_str, target_str, device=target_inp.device), dim=1) == entail_label_id
    para_score = para_score.view(batch_size, sample_size).float()


    if eval:
        return (para_score * ((1 - bleu_score) ** 2)), para_score, bleu_score
    return (para_score * ((1 - bleu_score) ** 2))