# reference: https://gist.github.com/danoneata/49a807f47656fedbb389
import numpy as np
from PQKNN import ProductQuantizationKNN
import time
import pandas as pd
import sys


# read a .ivecs file into numpy array
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


# read a .fvecs file into numpy array
def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

# calculate subvector centroids memory allocation
def get_size(subvector_centroids):
    mem = 0
    for arr in subvector_centroids.values():
        mem += arr.nbytes
    return mem

if __name__ == '__main__':
    # print vector space of all files in the siftsmall dataset
    small = True

    if small == True:
        base = fvecs_read("../siftsmall/siftsmall_base.fvecs")
        groundtruth = ivecs_read("../siftsmall/siftsmall_groundtruth.ivecs")
        learn = fvecs_read("../siftsmall/siftsmall_learn.fvecs")
        query = fvecs_read("../siftsmall/siftsmall_query.fvecs")
    else:
        base = fvecs_read("../sift/sift_base.fvecs")
        groundtruth = ivecs_read("../sift/sift_groundtruth.ivecs")
        learn = fvecs_read("../sift/sift_learn.fvecs")
        query = fvecs_read("../sift/sift_query.fvecs")

    # Create PQKNN object that partitions each train sample in n subvectors and c determines the amount of centroids for KMeans (2^c).
    # number of dimensions in dataset should be divisible by n (128 % n == 0); larger c -> higher accuracy
    metrics = []

    for n in [4, 8, 16, 32]:
        for c in [8, 9, 10, 11]:
            if small == True:
                log_file_name = "hyperparameter_tuning_logs_siftsmall.txt"
            else:
                log_file_name = "hyperparameter_tuning_logs_sift.txt"
            with open(log_file_name, "a") as log_file:
                print(f'n = {n}, c = {c}')
                log_file.write(f'n = {n}, c = {c}\n')
                # Initialize
                pqknn = ProductQuantizationKNN(n=n, c=c)

                # Perform the compression
                start_compression = time.time()
                pqknn.compress(base, np.arange(0, base.shape[0]))
                end_compression = time.time()
                print('Compressing the base vectors took',
                      (end_compression - start_compression), 'seconds.')
                log_file.write(f'Compressing the base vectors took {end_compression - start_compression} seconds.\n')
                log_file.write(f'Compressed data bytes: {pqknn.compressed_data.nbytes}')
                log_file.write(f'Subvector centroids bytes: {get_size(pqknn.subvector_centroids)}')

                # Find k-Nearest Neighbor search (with k = 100 - depending on dataset) for test data with the compressed training
                start_prediction = time.time()
                preds = pqknn.predict(query, nearest_neighbors=100)
                end_prediction = time.time()
                print('Predicting the', query.shape,
                      'query took', (end_prediction - start_prediction), 'seconds.')
                log_file.write(
                    f'Predicting the {query.shape} query took {end_prediction - start_prediction} seconds.\n')

                # Calculate recall (non-index based)
                avg = []
                for j in range(query.shape[0]):
                    avg.append(
                        np.mean([1 if i in groundtruth[j] else 0 for i in preds[j]]))
                print(f'recall = {np.mean(avg)}\n\n')
                log_file.write(f'recall = {np.mean(avg)}\n\n')
                metrics.append([n, c, end_compression - start_compression,
                               end_prediction - start_prediction, pqknn.compressed_data.nbytes, get_size(pqknn.subvector_centroids), np.mean(avg)])
            log_file.close()
    metrics_df = pd.DataFrame(
        metrics, columns=["n", "c", "compression_time", "prediction_time", "compression_bytes", "subvector_centroids_bytes", "recall"])
    if small == True:
        metrics_df.to_csv("df_metrics_siftsmall.csv", index=False)
    else:
        metrics_df.to_csv("df_metrics_sift.csv", index=False)
