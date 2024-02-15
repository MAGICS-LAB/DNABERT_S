import numpy as np
import transformers
import torch
import torch.utils.data as util_data
import torch.nn as nn
import tqdm
import os

from scipy.optimize import linear_sum_assignment

def get_embedding(dna_sequences, 
                  model, 
                  species, 
                  sample, 
                  task_name="clustering",
                  post_fix="",
                  test_model_dir="./test_model"):
    model2filename = {
        "tnf": "tnf.npy",
        "tnf_k": "tnf_k.npy",
        "dna2vec": "dna2vec.npy",
        "hyenadna": "hyenadna.npy",
        "dnabert2": "dnabert2_new.npy",
        "nt": "nt.npy",
        "test": "test.npy",
    }
    
    model2batch_size = {
        "tnf": 100,
        "tnf_k": 100,
        "dna2vec": 100, 
        "hyenadna": 100,
        "dnabert2": 20,
        "nt": 64,
        "test": 20,
    }
    batch_size = model2batch_size[model]
    
    embedding_dir = f"embeddings/{species}/{task_name}_{sample}{post_fix}"
    embedding_file = os.path.join(embedding_dir, model2filename[model])
    if os.path.exists(embedding_file):
        print(f"Load embedding from file {embedding_file}")
        embedding = np.load(embedding_file)
    
    else:
        print(f"Calculate embedding for {model} {species} {sample}")
        
        if model == "tnf":
            embedding = calculate_tnf(dna_sequences)
        elif model == "tnf_k":
            embedding = calculate_tnf(dna_sequences, kernel=True)
        elif model == "dna2vec":
            embedding = calculate_dna2vec_embedding(dna_sequences, 
                                                    embedding_dir=embedding_dir,)
        elif model == "hyenadna":
            embedding = calculate_llm_embedding(dna_sequences, 
                                                model_name_or_path="LongSafari/hyenadna-medium-450k-seqlen-hf", 
                                                model_max_length=20000,
                                                batch_size=batch_size,)
        elif model == "dnabert2":
            embedding = calculate_llm_embedding(dna_sequences,
                                                model_name_or_path="zhihan1996/DNABERT-2-117M", 
                                                model_max_length=5000,
                                                batch_size=batch_size,)
        elif model == "nt":
            embedding = calculate_llm_embedding(dna_sequences, 
                                                model_name_or_path="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", 
                                                model_max_length=2048,
                                                batch_size=batch_size,)
        elif model == "test":
            embedding = calculate_llm_embedding(dna_sequences, 
                                                model_name_or_path=test_model_dir, 
                                                model_max_length=5000,
                                                batch_size=batch_size,)
        else:
            raise ValueError(f"Unknown model {model}")
        
        if model not in ["bpe", "dna2vec"]:
            print(f"Save embedding to file {embedding_file}")
            os.makedirs(embedding_dir, exist_ok=True)
            np.save(embedding_file, embedding)
        
    return embedding


def calculate_tnf(dna_sequences, kernel=False):
    # Define all possible tetra-nucleotides
    nucleotides = ['A', 'T', 'C', 'G']
    tetra_nucleotides = [a+b+c+d for a in nucleotides for b in nucleotides for c in nucleotides for d in nucleotides]
    
    # build mapping from tetra-nucleotide to index
    tnf_index = {tn: i for i, tn in enumerate(tetra_nucleotides)}        

    # Iterate over each sequence and update counts
    embedding = np.zeros((len(dna_sequences), len(tetra_nucleotides)))
    for j, seq in enumerate(dna_sequences):
        for i in range(len(seq) - 3):
            tetra_nuc = seq[i:i+4]
            embedding[j, tnf_index[tetra_nuc]] += 1
    
    # Convert counts to frequencies
    total_counts = np.sum(embedding, axis=1)
    embedding = embedding / total_counts[:, None]

    if kernel:
        def validate_input_array(array):
            "Returns array similar to input array but C-contiguous and with own data."
            if not array.flags["C_CONTIGUOUS"]:
                array = np.ascontiguousarray(array)
            if not array.flags["OWNDATA"]:
                array = array.copy()

            assert array.flags["C_CONTIGUOUS"] and array.flags["OWNDATA"]

            return array

        npz = np.load("./helper/kernel.npz")
        kernel = validate_input_array(npz["arr_0"])
        embedding += -(1 / 256)
        embedding = np.dot(embedding, kernel)
        
    return embedding


def calculate_dna2vec_embedding(dna_sequences, embedding_dir):
    embedding_file = os.path.join(embedding_dir, "tnf.npy")
    if os.path.exists(embedding_file):
        print(f"Load embedding from file {embedding_file}")
        tnf_embedding = np.load(embedding_file)
    else:
        tnf_embedding = calculate_tnf(dna_sequences)
        
    kmer_embedding = np.load("./helper/4mer_embedding.npy")
    # kmer_embedding = np.random.normal(size=(256, 100))
    
    embedding = np.dot(tnf_embedding, kmer_embedding)    
    
    return embedding




def calculate_llm_embedding(dna_sequences, model_name_or_path, model_max_length=400, batch_size=20):
    # reorder the sequences by length
    lengths = [len(seq) for seq in dna_sequences]
    idx = np.argsort(lengths)
    dna_sequences = [dna_sequences[i] for i in idx]
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=None,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

    is_hyenadna = "hyenadna" in model_name_or_path
    is_nt = "nucleotide-transformer" in model_name_or_path
    
    if is_nt:
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        ) 
    else:
        model = transformers.AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
    

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = nn.DataParallel(model)
        
    model.to("cuda")


    train_loader = util_data.DataLoader(dna_sequences, batch_size=batch_size*n_gpu, shuffle=False, num_workers=2*n_gpu)
    for j, batch in enumerate(tqdm.tqdm(train_loader)):
        with torch.no_grad():
            token_feat = tokenizer.batch_encode_plus(
                    batch, 
                    max_length=model_max_length, 
                    return_tensors='pt', 
                    padding='longest', 
                    truncation=True
                )
            input_ids = token_feat['input_ids'].cuda()
            attention_mask = token_feat['attention_mask'].cuda()
            if is_hyenadna:
                model_output = model.forward(input_ids=input_ids)[0].detach().cpu()
            else:
                model_output = model.forward(input_ids=input_ids, attention_mask=attention_mask)[0].detach().cpu()
                
            attention_mask = attention_mask.unsqueeze(-1).detach().cpu()
            embedding = torch.sum(model_output*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
            
            if j==0:
                embeddings = embedding
            else:
                
                embeddings = torch.cat((embeddings, embedding), dim=0)

    embeddings = np.array(embeddings.detach().cpu())
    
    # reorder the embeddings
    embeddings = embeddings[np.argsort(idx)]

    return embeddings

def KMedoid(features,
            min_similarity=0.8,
            min_bin_size=100,
            max_iter=300):
    # rank nodes by the number of neighbors
    features = features.astype(np.float32)
    similarities = np.dot(features, features.T)

    # set the values below min_similarity to 0
    similarities[similarities < min_similarity] = 0
    row_sum = np.sum(similarities, axis=1)

    labels = np.ones(len(features)) * -1
    labels = labels.astype(int)
    count = 0

    while np.any(labels == -1):
        count += 1
        if count > max_iter:
            break
        # i = np.random.choice(np.where(labels == -1)[0])
        i = np.argmax(row_sum)
        # print(f"i: {i} count: {count} row_sum: {row_sum[i]}")
        row_sum[i] = -100

        medoid = features[i]
        idx_within = np.zeros(len(features), dtype=bool)
        idx_available = labels == -1

        for _ in range(3):
            similarity = np.dot(features, medoid)
            idx_within = similarity >= min_similarity
            # idx_within = np.logical_or(idx_within, similarity >= min_similarity)
            idx = np.where(np.logical_and(idx_within, idx_available))[0]
            medoid = np.mean(features[idx], axis=0)

        # assign labels
        labels[idx] = count
        row_sum -= np.sum(similarities[:, idx], axis=1)
        row_sum[idx] = -100
        
    
    # remove bins that are too small
    unique, counts = np.unique(labels, return_counts=True)
    for i, c in zip(unique, counts):
        if c < min_bin_size:
            labels[labels == i] = -1
    
    return labels



def align_labels_via_hungarian_algorithm(true_labels, predicted_labels):
    """
    Aligns the predicted labels with the true labels using the Hungarian algorithm.

    Args:
    true_labels (list or array): The true labels of the data.
    predicted_labels (list or array): The labels predicted by a clustering algorithm.

    Returns:
    dict: A dictionary mapping the predicted labels to the aligned true labels.
    """
    # Create a confusion matrix
    max_label = max(max(true_labels), max(predicted_labels)) + 1
    confusion_matrix = np.zeros((max_label, max_label), dtype=int)

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[true_label, predicted_label] += 1

    # Apply the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(confusion_matrix, maximize=True)

    # Create a mapping from predicted labels to true labels
    label_mapping = {predicted_label: true_label for true_label, predicted_label in zip(row_ind, col_ind)}

    return label_mapping



def compute_class_center_medium_similarity(embeddings, labels):
    idx = np.argsort(labels)
    embeddings = embeddings[idx]
    labels = labels[idx]
    n_sample_per_class = np.bincount(labels)
        
    all_similarities = np.zeros(len(embeddings))
    count = 0
    
    for i in range(len(n_sample_per_class)):
        start = count
        end = count + n_sample_per_class[i]
        mean = np.mean(embeddings[start:end], axis=0)
        similarities = np.dot(mean, embeddings[start:end].T).reshape(-1)
        
        all_similarities[start:end] = similarities
        
        count += n_sample_per_class[i]
    
    all_similarities.sort()
    percentile_values = []
    for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        value = all_similarities[int(percentile/100 * len(embeddings))]
        percentile_values.append(value)
    print(percentile_values)
    
    
    return percentile_values



