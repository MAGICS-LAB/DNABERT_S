import os
import csv
import torch.utils.data as util_data
from torch.utils.data import Dataset

class PairSamples(Dataset):
    def __init__(self, train_x1, train_x2, pairsimi):
        assert len(pairsimi) == len(train_x1) == len(train_x2)
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.pairsimi = pairsimi
        
    def __len__(self):
        return len(self.pairsimi)

    def __getitem__(self, idx):
        return {'seq1': self.train_x1[idx], 'seq2': self.train_x2[idx], 'pairsimi': self.pairsimi[idx]}

'''
Assumed data format:
DNA sequence1, DNA sequence2
'''

def pair_loader_csv(args, load_train=True):
    delimiter = ","
    if load_train:
        with open(os.path.join(args.datapath, args.train_dataname)) as csvfile:
            data = list(csv.reader(csvfile, delimiter=delimiter))
    else:
        with open(os.path.join(args.datapath, args.val_dataname)) as csvfile:
            data = list(csv.reader(csvfile, delimiter=delimiter))
    if args.con_method=="same_species":
        seq1 = [d[0] for d in data] 
        seq2 = [d[1] for d in data] 
    else:
        seq1 = [d[0] for d in data] + [d[1] for d in data]
        seq2 = seq1
    pairsimi = [1 for _ in seq1]

    dataset = PairSamples(seq1, seq2, pairsimi)
    if load_train:
        loader = util_data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    else:
        loader = util_data.DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=4)
    return loader