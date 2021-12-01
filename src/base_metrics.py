# reference: https://gist.github.com/danoneata/49a807f47656fedbb389
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time

# read a .ivecs file into numpy array


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

# read a .fvecs file into numpy array


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def main():
    # print vector space of all files in the siftsmall dataset
    base = fvecs_read("../siftsmall/siftsmall_base.fvecs")

    groundtruth = ivecs_read("../siftsmall/siftsmall_groundtruth.ivecs")

    learn = fvecs_read("../siftsmall/siftsmall_learn.fvecs")

    query = fvecs_read("../siftsmall/siftsmall_query.fvecs")

    # Find k-Nearest Neighbor search (with k = 100) for test data with the compressed training
    neigh = KNeighborsClassifier(n_neighbors=100)
    neigh.fit(base, np.arange(0, base.shape[0]))
    start_prediction = time.time()
    preds = []
    for i in range(len(query)):
        preds.append(neigh.predict([query[i]]))
    end_prediction = time.time()
    print('Predicting the', query.shape,
          'query took', (end_prediction - start_prediction), 'seconds.')

    # Calculate recall (non-index based)
    avg = []
    for j in range(query.shape[0]):
        avg.append(
            np.mean([1 if i in groundtruth[j] else 0 for i in preds[j]]))
    print(f'recall: {np.mean(avg)}')
    print(f'memory: {base.nbytes}')


if __name__ == '__main__':
    main()
