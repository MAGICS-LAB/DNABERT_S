# Scripts for our experiments

export PATH_TO_RESULT_DICT=/path/to/result/dict # (e.g., ./results)
export PATH_TO_DATA_DICT=/path/to/data/dict # (e.g., /root/data/DNABERT_s_data)

# 1. Main Experiment:

# Curriculum contrastive learning (DNABERT-S): Weighted SimCLR + Manifold i-Mix
python main.py \
    --resdir PATH_TO_RESULT_DICT \
    --datapath PATH_TO_DATA_DICT \
    --train_dataname train_2w.csv \
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
    --curriculum \

# 2. Baseline Experiments

# (1). DNA-Dropout: Contrastive learning with dropout method
python main.py \
    --resdir PATH_TO_RESULT_DICT \
    --datapath PATH_TO_DATA_DICT \
    --train_dataname train_2w.csv \
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
    --con_method dropout \

# (2). DNA-Double: Contrastive learning with double strand method
python main.py \
    --resdir PATH_TO_RESULT_DICT \
    --datapath PATH_TO_DATA_DICT \
    --train_dataname train_2w.csv \
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
    --con_method double_strand \

# (3). DNA-Mutate: Contrastive learning with mutate method
python main.py \
    --resdir PATH_TO_RESULT_DICT \
    --datapath PATH_TO_DATA_DICT \
    --train_dataname train_2w.csv \
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
    --con_method mutate \

# 3. Ablation Experiments

# (1). Weighted SimCLR + i-Mix, without Manifold method
python main.py \
    --resdir PATH_TO_RESULT_DICT \
    --datapath PATH_TO_DATA_DICT \
    --train_dataname train_2w.csv \
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
    --mix_layer_num 0 \
    --curriculum \

# (2). Only Weighted SimCLR
python main.py \
    --resdir PATH_TO_RESULT_DICT \
    --datapath PATH_TO_DATA_DICT \
    --train_dataname train_2w.csv \
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

# (3). Only Manifold i-Mix
python main.py \
    --resdir PATH_TO_RESULT_DICT \
    --datapath PATH_TO_DATA_DICT \
    --train_dataname train_2w.csv \
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

# (4). Only i-Mix
python main.py \
    --resdir PATH_TO_RESULT_DICT \
    --datapath PATH_TO_DATA_DICT \
    --train_dataname train_2w.csv \
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
    --mix_layer_num 0 \

# 4. Other experiments in Appendix

# Experiments for Impact of Sequence Length

# (1). Only Weighted SimCLR for sequence len 500, max tokenization length = 100
python main.py \
    --resdir PATH_TO_RESULT_DICT \
    --datapath PATH_TO_DATA_DICT \
    --train_dataname train_2w.csv \
    --val_dataname val_48k.csv \
    --seed 1 \
    --logging_step 10000 \
    --logging_num 12 \
    --max_length 100 \
    --train_batch_size 48 \
    --val_batch_size 360 \
    --lr 3e-06 \
    --lr_scale 100 \
    --epochs 3 \
    --feat_dim 128 \
    --temperature 0.05 \
    --con_method same_species \

# (2). Only Weighted SimCLR for sequence len 2000, max tokenization length = 400
python main.py \
    --resdir PATH_TO_RESULT_DICT \
    --datapath PATH_TO_DATA_DICT \
    --train_dataname train_2w.csv \
    --val_dataname val_48k.csv \
    --seed 1 \
    --logging_step 10000 \
    --logging_num 12 \
    --max_length 400 \
    --train_batch_size 48 \
    --val_batch_size 360 \
    --lr 3e-06 \
    --lr_scale 100 \
    --epochs 3 \
    --feat_dim 128 \
    --temperature 0.05 \
    --con_method same_species \