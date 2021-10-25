# reference: https://gist.github.com/danoneata/49a807f47656fedbb389
import numpy as np
from PQKNN import ProductQuantizationKNN
import sklearn

# read a .ivecs file into numpy array


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

# read a .fvecs file into numpy array


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


if __name__ == '__main__':
    # print vector space of all files in the siftsmall dataset
    base = fvecs_read("siftsmall/siftsmall_base.fvecs")

    groundtruth = ivecs_read("siftsmall/siftsmall_groundtruth.ivecs")

    learn = fvecs_read("siftsmall/siftsmall_learn.fvecs")

    query = fvecs_read("siftsmall/siftsmall_query.fvecs")

    # Create PQKNN object that partitions each train sample in 7 subvectors and encodes each subvector in 7 bits.
    pqknn = ProductQuantizationKNN(n=7, c=7)
    # Perform the compression
    pqknn.compress(base, np.arange(0, base.shape[0]))

    # Classify the test data using k-Nearest Neighbor search (with k = 10) on the compressed training
    preds = pqknn.predict(query, nearest_neighbors=100)

    # print(sklearn.metrics.recall_score(
    # list(groundtruth[0]), list(preds[0]), average='micro'))
    avg = []
    for j in range(100):
        avg.append(np.sum([1 for i in preds[j] if i in groundtruth[j]]))
        #print(np.sum([1 for i in preds[j] if i in groundtruth[j]]))
        '''print(sklearn.metrics.recall_score(
            list(groundtruth[j]), list(preds[j]), average='micro'))'''
    print(np.sum(avg)/100)
