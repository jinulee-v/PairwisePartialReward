"""
"""
from tqdm import tqdm

import torch
import os
from evaluate import load
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
# from comet import download_model, load_from_checkpoint
from statistics import NormalDist

from functools import wraps
import logging 
def suspend_logging(func):
    @wraps(func)
    def inner(*args, **kwargs):
        logging.disable(logging.FATAL)
        try:
            return func(*args, **kwargs)
        finally:
            logging.disable(logging.NOTSET)
    return inner


batch_size=32

device = torch.device("cpu")
# BERT-score
metric_bert_score = load("bertscore")
bert_score_kwargs = {
    "model_type": "microsoft/deberta-large-mnli",
    "device": 'cuda:0',
}
# BLEU
metric_bleu = load("bleu")
metric_sacrebleu = load("sacrebleu")
sacrebleu_kwargs = {
    "tokenize": "intl"
}
# BERT-iBLEU
beta = 4.0
# BLEURT
bleurt_model = None
bleurt_tokenizer = None
bleurt_kwargs = {}
BLEURT_MODEL_ID = 'lucadiliello/BLEURT-20-D12'
# COMET
# comet_model = None
# comet_kwargs = {
#     "batch_size": 16,
#     "progress_bar": False
# }
# COMET_CACHE_PATH = os.path.expanduser("~/.comet/")
# COMET_MODEL_ID = "eamt22-cometinho-da"

def set_gpu(gpu=0):
    global device
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    bert_score_kwargs["device"] = str(device)

@torch.no_grad()
def get_bert_ibleu_score(inputs, _, samples, shape=None, extended=False, eval=False):
    """
    Metric for paraphrase generation (`--task paragen`).
    """
    global metric_bert_score
    if metric_bert_score is None:
        metric_bert_score = load("bertscore")

    if shape is not None:
        sample_n, beam_size = shape
    else:
        sample_n = len(inputs)
        beam_size = len(samples[0])

    if not extended:
        # assert len(inputs) == len(samples)
        extended_inputs = []
        extended_samples = []
        for t, s in zip(inputs, samples):
            # Prevent zero-division in BLEU score calculation
            t = t.strip() if len(t.strip()) > 0 else "."
            s = [(string.strip() if len(string.strip()) > 0 else ".") for string in s]
            extended_inputs.extend([[t]] * beam_size)
            extended_samples.extend(s)
        assert len(extended_inputs) == len(extended_samples)
    else:
        extended_inputs = inputs
        extended_samples = samples

    # BLEU score
    bleu_score = [metric_bleu.compute(predictions=[s if s.strip() else "."], references=[t])["bleu"] for s, t in tqdm(zip(extended_samples, extended_inputs))]
    bleu_score = torch.tensor(bleu_score).reshape((sample_n, beam_size)).to(device)
    ibleu_score = 1 - bleu_score

    # bert_score
    bert_score = metric_bert_score.compute(predictions=extended_samples, references=extended_inputs, **bert_score_kwargs)["f1"]
    bert_score = torch.tensor(bert_score).reshape((sample_n, beam_size)).to(device)

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

@torch.no_grad()
@suspend_logging
def get_bleurt_score(_, targets, samples, eval=False):
    """
    Metric for machine translation (`--task translation`).
    """
    global bleurt_model, bleurt_tokenizer
    if bleurt_model is None:
        bleurt_model = BleurtForSequenceClassification.from_pretrained(BLEURT_MODEL_ID)
        bleurt_model = bleurt_model.to(device).eval()
        bleurt_tokenizer = BleurtTokenizer.from_pretrained(BLEURT_MODEL_ID)

    assert len(targets) == len(samples)
    sample_n = len(targets)
    beam_size = len(samples[0])

    extended_targets = []
    extended_samples = []
    for t, s in zip(targets, samples):
        # Prevent zero-division in BLEU score calculation
        t = t.strip() if len(t.strip()) > 0 else "."
        s = [(string.strip() if len(string.strip()) > 0 else ".") for string in s]
        extended_targets.extend([t] * beam_size)
        extended_samples.extend(s)
    assert len(extended_targets) == len(extended_samples)

    # BLEU score
    res = []
    for start in range(0, len(extended_targets), batch_size):
        end = min(start + batch_size, len(extended_targets))
        inputs = bleurt_tokenizer(extended_targets[start:end], extended_samples[start:end], padding='longest', return_tensors='pt')
        inputs = inputs.to(device)
        res.extend(bleurt_model(**inputs).logits.flatten().tolist())

    bleurt_score = torch.tensor(res).reshape((sample_n, beam_size)).to(device)
    bleurt_score = torch.clamp(bleurt_score, 0, 1) # Normalize to 0~1 scale
    
    return bleurt_score

# @torch.no_grad()
# @suspend_logging
# def get_comet_score(inputs, targets, samples, eval=False):
#     """
#     Metric for machine translation (`--task translation`).
#     To rescale Z-score from COMET into 0-1 based, we apply CDF(z-score to cumulative probs).
#     """
#     global comet_model, comet_kwargs

#     # init comet model
#     if comet_model is None:
#         # comet_cache_path = os.path.join(COMET_CACHE_PATH, COMET_MODEL_ID, "checkpoints/model.ckpt")
#         comet_cache_path = download_model("Unbabel/wmt20-comet-da")
#         comet_model = load_from_checkpoint(comet_cache_path)
        
#     assert len(inputs) == len(samples)
#     sample_n = len(inputs)
#     beam_size = len(samples[0])

#     # Reformat
#     comet_inputs = []
#     for i, t, s in zip(inputs, targets, samples):
#         # Prevent zero-division in BLEU score calculation
#         i = i.strip() if len(i.strip()) > 0 else "."
#         t = t.strip() if len(t.strip()) > 0 else "."
#         s = [(string.strip() if len(string.strip()) > 0 else ".") for string in s]
#         for sample in s:
#             comet_inputs.append({
#                 "src": i,
#                 "ref": t,
#                 "mt": sample
#             })
#     model_output = comet_model.predict(comet_inputs, **comet_kwargs)[0]
#     nd = NormalDist()
#     model_output = [nd.cdf(val) for val in model_output] # Convert z-score to 0-1 prob
#     assert len(model_output) == sample_n * beam_size
#     model_output = torch.tensor(model_output, dtype=torch.float32).reshape((sample_n, beam_size))

#     return model_output


if __name__ == "__main__":
    # Example for different metrics
    source = ["Hola, mucho gusto."]
    target = ["Hello, nice to meet you."]
    samples = [["Hello, nice to meet you.", "Hello, nice to see you.", "Greetings, it is good to see you."]]
    print(get_bert_ibleu_score(target, None, samples))
    print(get_bleu_score(None, target, samples))
    print(get_bleurt_score(None, target, samples))
    # print(get_comet_score(source, target, samples))