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
            from_idx, to_idx = tuple(list(torch.randperm(len(sentences))[:2]))
            return sentences[from_idx], sentences[to_idx]
        else:
            return sentences[0], sentences[1]

class ParaphraseGenerationEvalDataset(ParaphraseGenerationDataset):
    """
    Dataset for BLEU evaluation
    """
    def __getitem__(self, i):
        sentences = self.data[i]["paraphrases"]
        return sentences[0], sentences[1:]

def pg_collate_fn(batch):
    froms, tos = [], []
    for f, t in batch:
        froms.append(f)
        tos.append(t)
    return froms, tos

class SynonymBranchingEvalDataset():
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        datum = self.data[i]
        return datum["input"], datum["output_prefix"], datum["original"], datum["synonym"]
