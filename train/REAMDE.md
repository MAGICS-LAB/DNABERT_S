
# 1. Setup environment

## 1.1 (Optional) Create conda environment
```
conda create -n DNABERT_S python=3.9
conda activate DNABERT_S
```
weimin
## 1.2 Install dependencies

```
pip install -r requirements.txt
pip uninstall triton # this can lead to errors in GPUs other than A100
```

# 2. Prepare data

Please first download the training data 

```
gdown 1p59ch_MO-9DXh3LUIvorllPJGLEAwsUp # pip install gdown
unzip dnabert-s_train.zip  # unzip the data 
```



# 3. Training

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
