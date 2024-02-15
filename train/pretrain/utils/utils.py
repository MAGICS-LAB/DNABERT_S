import os
import random
import torch
import numpy as np

def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def setup_path(args):
    resPath = f'epoch{args.epochs}'
    resPath += f'.{args.train_dataname}'
    resPath += f'.lr{args.lr}'
    resPath += f'.lrscale{args.lr_scale}'
    resPath += f'.bs{args.train_batch_size}'
    resPath += f'.maxlength{args.max_length}'
    resPath += f'.tmp{args.temperature}'
    resPath += f'.seed{args.seed}'
    resPath += f'.con_method{args.con_method}'
    resPath += f'.mix{args.mix}'
    resPath += f'.mix_layer_num{args.mix_layer_num}'
    resPath += f'.curriculum{args.curriculum}/'
    resPath = args.resdir + resPath
    print(f'results path: {resPath}')

    return resPath
