# DNABERT_S: Learning Species-Aware DNA Embedding with Genome Foundation Models



This Repo is the official implementatation of [DNABERT_S: Learning Species-Aware DNA Embedding with Genome Foundation Models](https://arxiv.org/abs/2402.08777).



## Contents

- [1. Introduction](#1-introduction)
- [2. Model and Data](#2-model-and-data)
- [3. Setup Environment](#3-setup-environment)
- [4. Quick Start](#4-quick-start)
- [5. Training](#5-training)
- [6. Evaluation](#6-evaluation)
- [7. Citation](#7-citation)



## 1. Introduction

DNABERT-S is a foundation model based on [DNABERT-2](https://github.com/Zhihan1996/DNABERT_2) specifically designed for generating DNA embedding that naturally clusters and segregates genome of different species in the embedding space, which can greatly benefit a wide range of genome applications, including species classification/identification, metagenomics binning, and understanding evolutionary relationships.



## 2. Model and Data



### 2.1 Model

The pre-trained models is available at Huggingface as `zhihan1996/DNABERT-S`.

To download the model from command line:

```
# command line
gdown 1p59ch_MO-9DXh3LUIvorllPJGLEAwsUp # pip install gdown
unzip dnabert-s_train.zip  # unzip the data 
```





### 2.2 Data

The training data of DNABERT-S is available at

```
gdown 1p59ch_MO-9DXh3LUIvorllPJGLEAwsUp # pip install gdown
unzip dnabert-s_train.zip  # unzip the data 
```



The evaluation data is available at

```
gdown 1I44T2alXrtXPZrhkuca6QP3tFHxDW98c # pip install gdown
unzip dnabert-s_eval.zip  # unzip the data 
```





## 3. Setup environment

```
conda create -n DNABERT_S python=3.9
conda activate DNABERT_S
```

```
pip install -r requirements.txt
pip uninstall triton # this can lead to errors in GPUs other than A100
```





## 4. Quick Start

Our model is easy to use with the [transformers](https://github.com/huggingface/transformers) package.


To load the model from huggingface:

```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
```


To calculate the embedding of a dna sequence

```
dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
hidden_states = model(inputs)[0] # [1, sequence_length, 768]

# embedding with mean pooling
embedding_mean = torch.mean(hidden_states[0], dim=0)
print(embedding_mean.shape) # expect to be 768
```



## 5. Training

Our code base expects pairs of DNA sequencesfor pre-training. We expect the training data to be a csv file with two columns and no header. Each row contains one pair of DNA sequences that you want to model to generate similar embedding for them. See data/debug_train.csv for an example.

Important arguments:

- resdir: dictionary to save model parameters
- datapath: dictionary of data
- train_dataname: the name of the training data file (e.g., "a.csv")
- val_dataname: the name of the validating data file (e.g., "a.csv")
- max_length: set it as 0.2 * DNA_length  (e.g., 200 for 1000-bp DNA)
- train_batch_size: batch size for training data, change it to fit your GPU RAM 
- con_method: contrastive learning method, including "same species", "dropout", "double_strand", "mutate"
- mix: whether use i-Mix method
- mix_layer_num: which layer to perform i-Mix, if the value is -1, it means manifold i-Mix
- curriculum: whether use curriculum learning
- Other arguments can also be adjusted.

For our curriculum contrastive learning method, you can use:

```
cd pretrain
export PATH_TO_DATA_DICT=/path/to/data
export TRAIN_FILE=debug_train.csv # use this for debug, for real training, please use train_2m.csv

python main.py \
    --resdir ./results/ \
    --datapath ${PATH_TO_DATA_DICT} \
    --train_dataname ${TRAIN_FILE} \
    --val_dataname val_48k.csv \
    --seed 1 \
    --logging_step 10000 \
    --logging_num 12 \
    --max_length 2000 \
    --train_batch_size 48 \
    --val_batch_size 360 \
    --lr 3e-06 \
    --lr_scale 100 \
    --epochs 3 \
    --feat_dim 128 \
    --temperature 0.05 \
    --con_method same_species \
    --mix \
    --mix_alpha 1.0 \
    --mix_layer_num -1 \
    --curriculum 
```

This training scripts expect 8 A100 80GB GPUs. If you are using other types of devices, please change the train_batch_size and max_length accordingly.

After model training, you will find the trained model at 
./pretrain/results/$file_name

The file_name is automatically set based on the hyperparameters, and the code regularly save checkpoint.

It should be something like ./results/contrastive.HardNeg.epoch2.debug_train.csv.lr3e-06.lrscale100.bs48.maxlength200.tmp0.05.decay1.seed1.turn1/100

The best model after validating is saved in ./pretrain/results/$file_name/best/

Scripts for other experiments are all in ./pretrain/results





## 6. Evaluation

### 6.1 Prepare model

```
cd evaluate
```





#### 6.1.1 Test pre-trained DNABERT-S

```
gdown 1ejNOMXdycorDzphLT6jnfGIPUxi6fO0g
unzip DNABERT-S.zip
export MODEL_DIR=/path/to/DNABERT-S (e.g., /root/Downloads/DNABERT-S)
```





#### 6.1.2 Test you own model train with our code base

Copy the necessary files to the folder where the model is saved. This is a bug in Huggingface Transformers package. Sometimes the model file such as `bert_layer.py` are not automatically saved to the model directory together with the model weights. So we manually do it.

```
export MODEL_DIR=/path/to/the/trained/model # (e.g., /root/ICML2024/train/pretrain/results/epoch3.debug_train.csv.lr3e-06.lrscale100.bs24.maxlength2000.tmp0.05.seed1.con_methodsame_species.mixTrue.mix_layer_num-1.curriculumTrue/0)

cp model_codes/* ${MODEL_DIR}
```



### 6.2 Clustering and Classification

```
export DATA_DIR=/path/to/the/unziped/folders

# evaluate the trained model
python eval_clustering_classification.py --test_model_dir ${MODEL_DIR} --data_dir ${DATA_DIR} --model_list "test"

# evaluate baselines (e.g., TNF and DNABERT-2)
python eval_clustering_classification.py --data_dir ${DATA_DIR} --model_list "tnf, dnabert2"
```





### 6.3 Metagenomics Binning

```
export DATA_DIR=/path/to/the/unziped/folders
export MODEL_DIR=/path/to/the/trained/model

# evaluate the trained model
python eval_binning.py --test_model_dir ${MODEL_DIR} --data_dir ${DATA_DIR} --model_list "test"

# evaluate baselines (e.g., TNF and DNABERT-2)
python eval_binning.py --data_dir ${DATA_DIR} --model_list "tnf, dnabert2"
```








## 7. Citation

If you have any question regarding our paper or codes, please feel free to start an issue or email Zhihan Zhou (zhihanzhou2020@u.northwestern.edu).



If you use DNABERT-S in your work, please consider cite our papers:



**DNABERT-S**

```
@misc{zhou2024dnaberts,
      title={DNABERT-S: Learning Species-Aware DNA Embedding with Genome Foundation Models}, 
      author={Zhihan Zhou and Winmin Wu and Harrison Ho and Jiayi Wang and Lizhen Shi and Ramana V Davuluri and Zhong Wang and Han Liu},
      year={2024},
      eprint={2402.08777},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN}
}
```



**DNABERT-2**

```
@misc{zhou2023dnabert2,
      title={DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome}, 
      author={Zhihan Zhou and Yanrong Ji and Weijian Li and Pratik Dutta and Ramana Davuluri and Han Liu},
      year={2023},
      eprint={2306.15006},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN}
}
```

**DNABERT**

```
@article{ji2021dnabert,
    author = {Ji, Yanrong and Zhou, Zhihan and Liu, Han and Davuluri, Ramana V},
    title = "{DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome}",
    journal = {Bioinformatics},
    volume = {37},
    number = {15},
    pages = {2112-2120},
    year = {2021},
    month = {02},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btab083},
    url = {https://doi.org/10.1093/bioinformatics/btab083},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/37/15/2112/50578892/btab083.pdf},
}
```

