from sklearn.preprocessing import normalize
import csv
import argparse
import os
import sys
import collections
import numpy as np
import sklearn.metrics


from utils import get_embedding, KMedoid, align_labels_via_hungarian_algorithm, compute_class_center_medium_similarity

csv.field_size_limit(sys.maxsize)
csv.field_size_limit(sys.maxsize)
max_length = 20000


def main(args):
    model_list = args.model_list.split(",")
    for model in model_list:
        for species in ["reference", "marine", "plant"]:
            for sample in ["5", "6"]:

                print(f"Start {model} {species} {sample} binning")
                
                ###### load clutsering data to compute similarity threshold
                clustering_data_file =  os.path.join(args.data_dir, species, f"clustering_0.tsv")
                with open(clustering_data_file, "r") as f:
                    reader = csv.reader(f, delimiter="\t")
                    data = list(reader)[1:]
                
                dna_sequences = [d[0][:max_length] for d in data]
                labels = [d[1] for d in data]
                
                # convert labels to numeric values  
                label2id = {l: i for i, l in enumerate(set(labels))}
                labels = np.array([label2id[l] for l in labels])
                num_clusters = len(label2id)
                print(f"Get {len(dna_sequences)} sequences, {num_clusters} clusters for")   

                # generate embedding
                embedding = normalize(get_embedding(dna_sequences, model, species, 0, task_name="clustering", test_model_dir=args.test_model_dir))
                percentile_values = compute_class_center_medium_similarity(embedding, labels)
                threshold = percentile_values[-3]
                print(f"threshold: {threshold}")
                
                
                
                ###### load binning data
                data_file =  os.path.join(args.data_dir, species, f"binning_{sample}.tsv")
                
                with open(data_file, "r") as f:
                    reader = csv.reader(f, delimiter="\t")
                    data = list(reader)[1:]

                dna_sequences = [d[0][:max_length] for d in data]
                labels_bin = [d[1] for d in data]
                
                # filter sequences with length < 2500
                filterd_idx = [i for i, seq in enumerate(dna_sequences) if len(seq) >= 2500]
                dna_sequences = [dna_sequences[i] for i in filterd_idx]
                labels_bin = [labels_bin[i] for i in filterd_idx]
                
                # filter sequences with low abundance labels (less than 10)
                label_counts = collections.Counter(labels_bin)
                filterd_idx = [i for i, l in enumerate(labels_bin) if label_counts[l] >= 10]
                dna_sequences = [dna_sequences[i] for i in filterd_idx]
                labels_bin = [labels_bin[i] for i in filterd_idx]

                # convert labels to numeric values  
                label2id = {l: i for i, l in enumerate(set(labels_bin))}
                labels_bin = np.array([label2id[l] for l in labels_bin])
                num_clusters = len(label2id)
                print(f"Get {len(dna_sequences)} sequences, {num_clusters} clusters")   

                # generate embedding
                embedding = get_embedding(dna_sequences, model, species, sample, task_name="binning")
                if len(embedding) > len(filterd_idx):
                    embedding = embedding[np.array(filterd_idx)]
                embedding_norm = normalize(embedding)
                # percentile_values = compute_within_class_similarity(embedding_norm, labels_bin)
                
                binning_results = KMedoid(embedding_norm, min_similarity=threshold, min_bin_size=10, max_iter=1000)
                print(len(np.unique(binning_results)))
                
                # Example usage
                true_labels_bin = labels_bin[binning_results != -1]
                predicted_labels = binning_results[binning_results != -1]

                # Align labels
                alignment_bin = align_labels_via_hungarian_algorithm(true_labels_bin, predicted_labels)
                predicted_labels_bin = [alignment_bin[label] for label in predicted_labels]

                # Calculate purity, completeness, recall, and ARI
                recall_bin = sklearn.metrics.recall_score(true_labels_bin, predicted_labels_bin, average=None, zero_division=0)
                nonzero_bin = len(np.where(recall_bin != 0)[0])
                recall_bin.sort()
                
                f1_bin = sklearn.metrics.f1_score(true_labels_bin, predicted_labels_bin, average=None, zero_division=0)
                f1_bin.sort()
                thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                recall_results = []
                f1_results = []
                for threshold in thresholds:
                    recall_results.append(len(np.where(recall_bin > threshold)[0]))
                    f1_results.append(len(np.where(f1_bin > threshold)[0]))
                
                print(f"f1_results: {f1_results}")
                print(f"recall_results: {recall_results} \n")
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clustering')
    parser.add_argument('--test_model_dir', type=str, default="/root/trained_model", help='Directory to save trained models to test')
    parser.add_argument('--model_list', type=str, default="test", help='List of models to evaluate, separated by comma. Currently support [tnf, tnf-k, dnabert2, hyenadna, nt, test]')
    parser.add_argument('--data_dir', type=str, default="/root/data", help='Data directory')
    args = parser.parse_args()
    main(args)