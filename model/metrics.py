"""
"""

import torch
from evaluate import load

batch_size=32

device = torch.device("cpu")
metric_bert_score = load("bertscore")
bert_score_kwargs = {
    "model_type": "microsoft/deberta-large-mnli",
    "device": str(device),
    "batch_size": batch_size
}
metric_bleu = load("bleu")
metric_sacrebleu = load("sacrebleu")
sacrebleu_kwargs = {
    "tokenize": "intl"
}
beta = 4.0


def set_gpu(gpu=0):
    global device, bert_score_kwargs
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    bert_score_kwargs["device"] = str(device)

@torch.no_grad()
def get_bert_ibleu_score(targets, _, samples, eval=False):
    """
    Metric for paraphrase generation (`--task paragen`).
    """
    assert len(targets) == len(samples)
    sample_n = len(targets)
    beam_size = len(samples[0])

    extended_targets = []
    extended_samples = []
    for t, s in zip(targets, samples):
        # Prevent zero-division in BLEU score calculation
        t = t.strip() if len(t.strip()) > 0 else "."
        s = [(string.strip() if len(string.strip()) > 0 else ".") for string in s]
        extended_targets.extend([[t]] * beam_size)
        extended_samples.extend(s)
    assert len(extended_targets) == len(extended_samples)

    # bert_score
    bert_score = metric_bert_score.compute(predictions=extended_samples, references=extended_targets, **bert_score_kwargs)["f1"]
    bert_score = torch.tensor(bert_score).reshape((sample_n, beam_size)).to(device)

    # BLEU score
    bleu_score = [metric_bleu.compute(predictions=[s], references=[t])["bleu"] for s, t in zip(extended_samples, extended_targets)]
    bleu_score = torch.tensor(bleu_score).reshape((sample_n, beam_size)).to(device)
    ibleu_score = 1 - bleu_score

    bert_ibleu_score = (1 + beta) * bert_score * ibleu_score / (beta * ibleu_score + bert_score) # Modified harmonic mean to prevent zero-division

    if eval:
        return bert_ibleu_score, bert_score, bleu_score
    else:
        return bert_ibleu_score

@torch.no_grad()
def get_bleu_score(_, targets, samples, eval=False):
    """
    Metric for machine translation (`--task translation`).
    """
    assert len(targets) == len(samples)
    sample_n = len(targets)
    beam_size = len(samples[0])

    extended_targets = []
    extended_samples = []
    for t, s in zip(targets, samples):
        # Prevent zero-division in BLEU score calculation
        t = t.strip() if len(t.strip()) > 0 else "."
        s = [(string.strip() if len(string.strip()) > 0 else ".") for string in s]
        extended_targets.extend([[t]] * beam_size)
        extended_samples.extend(s)
    assert len(extended_targets) == len(extended_samples)

    # BLEU score
    bleu_score = [metric_sacrebleu.compute(predictions=[s], references=[t])["score"] for s, t in zip(extended_samples, extended_targets)]
    bleu_score = torch.tensor(bleu_score).reshape((sample_n, beam_size)).to(device) / 100 # Normalize to 0~1 scale
    
    return bleu_score

if __name__ == "__main__":
    # Example for BERT-iBLEU score calculation
    print(get_bert_ibleu_score(
        ["Hello, nice to meet you."],
        [["Hello, nice to meet you.", "Hello, nice to see you.", "Greetings, it is good to see you."]],
    eval=True))