import os 
import sys
sys.path.append( './' )
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import json
import numpy as np
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from training import Trainer
from dataloader.dataloader import pair_loader_csv
from utils.utils import set_global_random_seed, setup_path
from utils.optimizer import get_optimizer
from models.dnabert_s import DNABert_S

def run(args):
    args.resPath = setup_path(args)
    set_global_random_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_id = torch.cuda.device_count()
    print("\t {} GPUs available to use!".format(device_id))

    '''
    We assume paired training data (e.g., DNA sequences in positive pairs with two columns) 
    is always saved in csv format.
    '''
    train_loader = pair_loader_csv(args, load_train=True)
    val_loader = pair_loader_csv(args, load_train=False)

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = DNABert_S(feat_dim=args.feat_dim, mix=args.mix, model_mix_dict=args.dnabert2_mix_dict, curriculum=args.curriculum)
    optimizer = get_optimizer(model, args)
    model = nn.DataParallel(model)
    model.to(device)
    
    # set up the trainer
    trainer = Trainer(model, tokenizer, optimizer, train_loader, val_loader, args)
    trainer.train()
    trainer.val()
    return None

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_instance', type=str, default='local')
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0], help="The list of gpuid. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=1, help="Random seed")
    parser.add_argument('--resdir', type=str, default="./results")
    parser.add_argument('--logging_step', type=int, default=10000, help="How many iteration steps to save the model checkpoint and loss value once")
    parser.add_argument('--logging_num', type=int, default=12, help="How many times to log totally")
    # Dataset
    parser.add_argument('--datapath', type=str, default='./data/reference_genome_links/', help="The dict of data")
    parser.add_argument('--train_dataname', type=str, default='train_2m.csv', help="Name of the data used for training")
    parser.add_argument('--val_dataname', type=str, default='val_48k.csv', help="Name of the data used for validating")
    # Training parameters
    parser.add_argument('--max_length', type=int, default=2000, help="Max length of tokens")
    parser.add_argument('--train_batch_size', type=int, default=48, help="Batch size used for training dataset")
    parser.add_argument('--val_batch_size', type=int, default=360, help="Batch size used for validating dataset")
    parser.add_argument('--lr', type=float, default=3e-06, help="Learning rate")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--epochs', type=int, default=3)
    # Contrastive learning
    parser.add_argument('--feat_dim', type=int, default=128, help="Dimension of the projected features for instance discrimination loss")
    parser.add_argument('--temperature', type=float, default=0.05, help="Temperature required by contrastive loss")
    parser.add_argument('--con_method', type=str, default='same_species', help="Which data augmentation method used, include dropout, double_strand, mutate, same_species")
    parser.add_argument('--mix', action="store_true", help="Whether use i-Mix method")
    parser.add_argument('--dnabert2_mix_dict', type=str, default="./DNABERT-2-117M-MIX", help="Dictionary of the modified code for DNABert-2 to perform i-Mix")
    parser.add_argument('--mix_alpha', type=float, default=1.0, help="Value of alpha to generate i-Mix coefficient")
    parser.add_argument('--mix_layer_num', type=int, default=-1, help="Which layer to perform i-Mix, if the value is -1, it means manifold i-Mix")
    parser.add_argument('--curriculum', action="store_true", help="Whether use curriculum learning")
    
    
    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None
    return args

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    run(args)




    


