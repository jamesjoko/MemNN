# reference: https://gist.github.com/danoneata/49a807f47656fedbb389
import numpy as np
from PQKNN import ProductQuantizationKNN
import time
from memory_profiler import profile

# read a .ivecs file into numpy array


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

# read a .fvecs file into numpy array


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


fp = open('../memory_logs/memlog_n32_c12.log', 'w+')


@profile(stream=fp)
def main():
    # print vector space of all files in the siftsmall dataset
    base = fvecs_read("../siftsmall/siftsmall_base.fvecs")

    groundtruth = ivecs_read("../siftsmall/siftsmall_groundtruth.ivecs")

    learn = fvecs_read("../siftsmall/siftsmall_learn.fvecs")

    query = fvecs_read("../siftsmall/siftsmall_query.fvecs")

    # Create PQKNN object that partitions each train sample in n subvectors and encodes each subvector in 2^c bits.
    # number of dimensions in dataset should be divisible by n (128 % n == 0); larger c -> higher accuracy
    pqknn = ProductQuantizationKNN(n=32, c=12)
    # Perform the compression
    pqknn.compress(base, np.arange(0, base.shape[0]))

    # Find k-Nearest Neighbor search (with k = 100) for test data with the compressed training
    start_prediction = time.time()
    preds = pqknn.predict(query, nearest_neighbors=100)
    end_prediction = time.time()
    print('Predicting the', query.shape,
          'query took', (end_prediction - start_prediction), 'seconds.')

    # Calculate recall (non-index based)
    avg = []
    for j in range(query.shape[0]):
        avg.append(
            np.mean([1 if i in groundtruth[j] else 0 for i in preds[j]]))
    print(np.mean(avg))


if __name__ == '__main__':
    main()
