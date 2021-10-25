#reference: https://gist.github.com/danoneata/49a807f47656fedbb389
import numpy as np

#read a .ivecs file into numpy array
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

#read a .fvecs file into numpy array
def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


#print vector space of all files in the siftsmall dataset
base = fvecs_read("siftsmall/siftsmall_base.fvecs")
print(base)
print(base.shape)

groundtruth = ivecs_read("siftsmall/siftsmall_groundtruth.ivecs")
print(groundtruth)
print(groundtruth.shape)

learn = fvecs_read("siftsmall/siftsmall_learn.fvecs")
print(learn)
print(learn.shape)

query = fvecs_read("siftsmall/siftsmall_query.fvecs")
print(query)
print(query.shape)