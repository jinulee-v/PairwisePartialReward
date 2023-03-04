import torch

class TextGenerationDataset():
    def __init__(self, data, shuffle=True):
        self.data = data
        self.shuffle = shuffle
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        datum = self.data[i]
        if self.shuffle:
            tgt_idx = torch.randint(len(datum["targets"]), (1,)).item()
            return datum["source"], datum["targets"][tgt_idx]
        else:
            return datum["source"], datum["targets"][0]

def tg_collate_fn(batch):
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
        return datum["input"], datum["output_prefix"].rstrip(), ' ' + datum["original"], ' ' + datum["synonym"]
