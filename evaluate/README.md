# Evaluation Codes for DNABERT-S

# 1. Setup environment
Please follow the environment setup in pretrain/README.md to setup the virtual conda environment.



# 2. Prepare data
Please first download the evaluation data 

```
gdown 1I44T2alXrtXPZrhkuca6QP3tFHxDW98c # pip install gdown
unzip dnabert-s_eval.zip  # unzip the data 
```

# 3. Prepare model
## 3.1 Test pre-trained DNABERT-S

```
gdown 1ejNOMXdycorDzphLT6jnfGIPUxi6fO0g
unzip DNABERT-S.zip
export MODEL_DIR=/path/to/DNABERT-S (e.g., /root/Downloads/DNABERT-S)
```





## 3.2 Test you own model train with our code base

Copy the necessary files to the folder where the model is saved. This is a bug in Huggingface Transformers package. Sometimes the model file such as `bert_layer.py` are not automatically saved to the model directory together with the model weights. So we manually do it.

```
export MODEL_DIR=/path/to/the/trained/model # (e.g., /root/ICML2024/train/pretrain/results/epoch3.debug_train.csv.lr3e-06.lrscale100.bs24.maxlength2000.tmp0.05.seed1.con_methodsame_species.mixTrue.mix_layer_num-1.curriculumTrue/0)

cp model_codes/* ${MODEL_DIR}
```


# 4. Clustering and Classification

```
export DATA_DIR=/path/to/the/unziped/folders

# evaluate the trained model
python eval_clustering_classification.py --test_model_dir ${MODEL_DIR} --data_dir ${DATA_DIR} --model_list "test"

# evaluate baselines (e.g., TNF and DNABERT-2)
python eval_clustering_classification.py --data_dir ${DATA_DIR} --model_list "tnf, dnabert2"
```


# 5. Metagenomics Binning

```
export DATA_DIR=/path/to/the/unziped/folders
export MODEL_DIR=/path/to/the/trained/model

# evaluate the trained model
python eval_binning.py --test_model_dir ${MODEL_DIR} --data_dir ${DATA_DIR} --model_list "test"

# evaluate baselines (e.g., TNF and DNABERT-2)
python eval_binning.py --data_dir ${DATA_DIR} --model_list "tnf, dnabert2"
```