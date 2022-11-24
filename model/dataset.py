import random
import torch

class ParaphraseGenerationDataset():
    def __init__(self, data, shuffle=True):
        self.data = data
        self.shuffle = shuffle
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        sentences = self.data[i]["paraphrases"]
        if self.shuffle:
            from_idx, to_idx = random.sample(range(len(sentences)), 2)
            return sentences[from_idx], sentences[to_idx]
        else:
            return sentences[0], sentences[1]

def pg_collate_fn(batch):
    froms, tos = [], []
    for f, t in batch:
        froms.append(f)
        tos.append(t)
    return froms, tos

class ParaphraseGenerationEvalDataset():
    """
    Dataset for BLEU evaluation
    """
    def __init__(self, data, shuffle=True):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        sentences = self.data[i]["paraphrases"]
        return sentences[0], sentences[1:]


class ParaphraseIdentificationDataset():
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        sentences = self.data[i]
        return sentences["sentence1"], sentences["sentence2"], sentences["is_duplicate"]

def pi_collate_fn(batch):
    froms, tos, labels = [], [], []
    for f, t, label in batch:
        froms.append(f)
        tos.append(t)
        labels.append(label)
    return froms, tos, torch.tensor(labels)