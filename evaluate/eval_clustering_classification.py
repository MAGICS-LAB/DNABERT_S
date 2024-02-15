def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse
import os
from sklearn.preprocessing import normalize
import csv
import sys
import numpy as np
import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


from utils import get_embedding

csv.field_size_limit(sys.maxsize)
csv.field_size_limit(sys.maxsize)

def main(args):
    model_list = args.model_list.split(",")
    for model in model_list:
        for species in ["reference", "marine", "plant"]:
            max_length = 10000 if species == "reference" else 20000
            for sample in [0, 1, 2, 3, 4]:
                if species == "reference" and sample > 1:
                    continue
                sample = str(sample)
                
                print(f"Start {model} {species} {sample} clustering")
                data_file = os.path.join(args.data_dir, species, f"clustering_{sample}.tsv")
                                
                with open(data_file, "r") as f:
                    reader = csv.reader(f, delimiter="\t")
                    data = list(reader)[1:]

                dna_sequences = [d[0][:max_length] for d in data]
                labels = [d[1] for d in data]

                # convert labels to numeric values  
                label2id = {l: i for i, l in enumerate(set(labels))}
                labels = np.array([label2id[l] for l in labels])
                num_clusters = len(label2id)
                print(f"Get {len(dna_sequences)} sequences, {num_clusters} clusters")   

                # generate embedding
                embedding = get_embedding(dna_sequences, model, species, sample, test_model_dir=args.test_model_dir)
                embedding_norm = normalize(embedding)
                embedding_standard = StandardScaler().fit_transform(embedding)
                
                random_seeds = [0, 1, 2, 3, 4]
                kmeans_results = np.zeros([len(random_seeds), 4])
                lr_results = np.zeros([len(random_seeds), 5])
                
                for random_seed in random_seeds:
                    ### perform k-means clustering
                    kmeans = KMeans(n_clusters=num_clusters,
                                    random_state=random_seed,
                                    max_iter=1000,
                                    init="random",
                                    n_init=3)
                    kmeans.fit(embedding_norm)
                    preds_clustering = kmeans.labels_


                    purity = sklearn.metrics.homogeneity_score(labels, preds_clustering)
                    completeness = sklearn.metrics.completeness_score(labels, preds_clustering)
                    ari = sklearn.metrics.adjusted_rand_score(labels, preds_clustering)
                    nmi = sklearn.metrics.normalized_mutual_info_score(labels, preds_clustering)
                    
                    kmeans_results[random_seed] = np.array([purity, completeness, ari, nmi])
                    
                    print(f"Kmeans purity: {purity} completeness: {completeness} ari: {ari} nmi: {nmi}")
                    

                    # perform few-shot classification
                    results = []
                    
                    # generate different train/test splits for each random seed
                    np.random.seed(random_seed)
                    permutation = np.random.permutation(len(labels))
                    labels = labels[permutation]
                    embedding_norm = embedding_norm[permutation]
                    embedding_standard = embedding_standard[permutation]
                    
                    for num_samples_per_class in [1, 2, 5, 10, 20]:
                        is_train = np.zeros(len(labels))
                        is_test = np.zeros(len(labels))
                        for i in range(num_clusters):
                            idx = np.where(labels == i)[0]
                            is_train[idx[:num_samples_per_class]] = 1
                            is_test[idx[-80:]] = 1
                        is_train = is_train.astype(bool)
                        is_test = is_test.astype(bool)
                        
                        embedding_train = embedding_standard[is_train]
                        embedding_test = embedding_standard[is_test]

                        # 1. Logistic Regression
                        lr = LogisticRegression(random_state=random_seed, 
                                                max_iter=3000, 
                                                n_jobs=64,
                                                solver="lbfgs",
                                                penalty="l2",
                                                C=0.5)
                        lr.fit(embedding_train, labels[is_train])
                        preds_lr = lr.predict(embedding_test)
                        preds_train_lr = lr.predict(embedding_train)
                        
                        f1_train = sklearn.metrics.f1_score(labels[is_train], preds_train_lr, average="macro", zero_division=0)
                        loss_train = sklearn.metrics.log_loss(labels[is_train], lr.predict_proba(embedding_train))
                                        
                        f1 = sklearn.metrics.f1_score(labels[is_test], preds_lr, average="macro", zero_division=0)
                        recall = sklearn.metrics.recall_score(labels[is_test], preds_lr, average="macro", zero_division=0)
                        precision = sklearn.metrics.precision_score(labels[is_test], preds_lr, average="macro", zero_division=0)
                        accuracy = sklearn.metrics.accuracy_score(labels[is_test], preds_lr)
                        results.append(f1)
                        print(f"LR {num_samples_per_class}  train f1: {f1_train} loss: {loss_train} f1: {f1} recall: {recall} precision: {precision} accuracy: {accuracy}")
                        
                    
                    lr_results[random_seed] = np.array(results)
                
                kmeans_results = kmeans_results.mean(axis=0)
                print(f"Kmeans purity: {kmeans_results[0]} completeness: {kmeans_results[1]} ari: {kmeans_results[2]} nmi: {kmeans_results[3]}")
                
                lr_results = lr_results.mean(axis=0)
                lr_results = ", ".join([str(round(r, 6)) for r in lr_results])
                print(f"LR 1: {lr_results[0]} 2: {lr_results[1]} 5: {lr_results[2]} 10: {lr_results[3]} 20: {lr_results[4]}")
                print(lr_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clustering')
    parser.add_argument('--test_model_dir', type=str, default="/root/trained_model", help='Directory to save trained models to test')
    parser.add_argument('--model_list', type=str, default="tnf, test", help='List of models to evaluate, separated by comma. Currently support [tnf, tnf-k, dnabert2, hyenadna, nt, test]')
    parser.add_argument('--data_dir', type=str, default="/root/data", help='Data directory')
    args = parser.parse_args()
    main(args)