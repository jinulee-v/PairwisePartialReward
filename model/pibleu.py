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

device = torch.device("cpu")
# hg_model_hub_name = "textattack/bert-base-uncased-QQP"
hg_model_hub_name = "domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector"
pi_tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
pi_model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)
pad_id = pi_tokenizer.pad_token_id
entail_label_id = 1


def set_gpu(gpu=0):
    global device, pi_model
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    pi_model = pi_model.to(device)


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
    for head in range(0, length, batch_size):
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

# def get_bleu_score(s1, s2):
#     scores = [sentence_bleu([x1.split()], x2.split(), weights=(0.25,0.25,0.25,0.25)) for x1, x2 in zip(s1, s2)]
#     return torch.tensor(scores)

def form_ngram(input_tensor, n):
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

def n_gram_precision(ref_tensor, sys_tensor, pad_id, n_gram=4):
    """
    Calculates n-gram precision with brevity penalty.

    ref_tensor: batch x seq_len1
    sys_tensor: batch x sample_num x seq_len2
    """
    # Determine batch size, sample count(=beam size), n-gram
    bsz, sample_num = sys_tensor.size(0), sys_tensor.size(1)
    n = min(min(n_gram, ref_tensor.size(-1)), sys_tensor.size(-1))

    # Generate masks
    ref_padding = (~(ref_tensor == pad_id)).float()
    ref_ngram_mask = torch.arange(0, ref_padding.size(1), device=ref_padding.device) * torch.ones_like(ref_padding)
    ref_ngram_mask = torch.where(
        ref_ngram_mask < (torch.sum(ref_padding, dim=-1, keepdims=True) - n + 1),
        ref_padding, torch.zeros_like(ref_padding)
    )[:, :ref_ngram_mask.size(-1) - n + 1]
    sys_padding = (~(sys_tensor == pad_id)).float()
    sys_ngram_mask = torch.arange(0, sys_padding.size(-1), device=sys_padding.device) * torch.ones_like(sys_padding)
    sys_ngram_mask = torch.where(
        sys_ngram_mask < (torch.sum(sys_padding, dim=-1, keepdims=True) - n + 1),
        sys_padding, torch.zeros_like(sys_padding)
    )[:, :, :sys_ngram_mask.size(-1) - n + 1]

    # Get n-grams
    ref_tensor = ref_tensor * ref_padding # mask out paddings
    sys_tensor = sys_tensor * sys_padding
    ref_tensor = ref_tensor[:, None, :].repeat(1, sample_num, 1) # readjust ref size to match sys
    input_tensor1_ngram = form_ngram(ref_tensor, n).float()
    input_tensor2_ngram = form_ngram(sys_tensor, n).float()  # batch x sample_num x seq_len-(n-1) x n

    # Calculate similarity matrix
    sim_matrix = (torch.norm( # Calculate L2 norm to find if N-gram in `sys`` is present in `ref``
        input_tensor2_ngram.unsqueeze(3) - input_tensor1_ngram.unsqueeze(2),
        p=2, dim=-1
    ) == 0.0).to(torch.float)
    # print(sim_matrix.size(), sys_ngram_mask.size(), ref_ngram_mask.size())
    sim_matrix *= sys_ngram_mask.unsqueeze(3) * ref_ngram_mask.unsqueeze(1).unsqueeze(2)
    sim_matrix = torch.sum(torch.max(sim_matrix, dim=-1).values, dim=-1)

    # Brevity penalty
    ref_len = torch.sum(ref_padding, dim=-1, keepdims=True)
    sys_len = torch.sum(sys_padding, dim=-1)
    bp = torch.exp(1 -(ref_len / sys_len))
    bp = torch.where(ref_len >= sys_len, bp, torch.ones_like(bp))

    return sim_matrix / torch.sum(sys_ngram_mask, dim=-1) * bp  # batch x sample_num

def torch_bleu(ref_tensor, sys_tensor, pad_id):
    # If short enough, do not batchify
    if ref_tensor.size(0) < 2 * batch_size:
        batch_bleu = 0
        for n_gram in range(1, 5):       
            batch_bleu += n_gram_precision(ref_tensor, sys_tensor, pad_id, n_gram)
        batch_bleu /= 4
        return batch_bleu

    # Divide by batch
    bleu = None
    for head in range(0, ref_tensor.size(0), batch_size):
        tail = min(head + batch_size, ref_tensor.size(0))

        ref = ref_tensor[head:tail]
        sys = sys_tensor[head:tail]

        batch_bleu = 0
        for n_gram in range(1, 5):       
            batch_bleu += n_gram_precision(ref, sys, pad_id, n_gram)
        batch_bleu /= 4
        if bleu is None:
            bleu = batch_bleu
        else:
            bleu = torch.cat([bleu, batch_bleu], dim=0)
    bleu = bleu.nan_to_num(nan=0, posinf=0, neginf=0)
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
        return (para_score * (1 - bleu_score)), para_score, bleu_score
    return (para_score * (1 - bleu_score))

if __name__ == "__main__":
    # Example for BLEU score calculation
    ref = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]])
    sys = torch.tensor([[
        [1, 2, 3, 4, 5, 0, 0, 0, 0],
        [1, 2, 3, 4, 0, 0, 0, 0, 0],
        [1, 2, 3, 4, 5, 6, 0, 0, 0],
        [2, 3, 4, 5, 6, 0, 0, 0, 0],
    ]])
    for n_gram in range(1, 5):
        print(n_gram, "- gram :", n_gram_precision(ref, sys, pad_id=0, n_gram=n_gram))
    print(torch_bleu(ref, sys, pad_id=0))

    # Example for PiBLEU score calculation
    print(get_pibleu_score(
        ["Hello, nice to meet you."],
        [["Hello, nice to meet you.", "Hello, nice to see you.", "Greetings, it is good to see you."]],
        tokenizer=pi_tokenizer
    ))