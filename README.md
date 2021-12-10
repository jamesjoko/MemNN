# CS 490LDA (Wang) Group 1 Project, Memory-Based Nearest Neighbor Search (MemNN)

## Members: Annie Wong, Alan Cao, James Joko, Jerry Chen, Jim He

Dataset is located [here](http://corpus-texmex.irisa.fr/)

Shared Google Drive link is [here](https://drive.google.com/drive/folders/165mXYLS_edvVUcastpUGYvLEfZOgaFF-?usp=sharing)

Evaluation/performance will be measured in recall: tp / (tp + fn). Recall measures the ability of the classifier to identify all of the positive samples.
A true positive is when a neighbor is identified in the correct i-th nearest position.
A false negative is when a neighbor is identified in either the incorrect i-th nearest position or is not a k-nearest neighbor.

The data used contains 4 files:

- base.fvecs: The vectors in which a search is performed
- groundtruth.ivecs: pre-computed k nearest neighbors. test labels/true values
- learn.fvecs: vectors to find the parameters involved in a particular method (we do not use this)
- query.fvecs: test set

For the 10k vectors data, the sizes of these arrays are (10000, 128), (100, 100), (25000, 128), and (100, 128) respectively.

For the 1M vectors data, the sizes of these arrays are (1000000, 128), (10000, 100), (100000, 128), and (10000, 128) respectively.

Link to download [sift_base](https://drive.google.com/file/d/1Hm0IUfnZhXwmdyFq0o28135WFuNBbPpB/view?usp=sharing) file (place in sift folder)

Memory is calculated by adding the nbytes attribute of the numpy arrays of the PQKNN object (compressed_data and subvector_centroids)

To collect metric data, run `python src/hyperparameter_tuning.py`. Logs will be found at `src/hyperparameter_tuning_logs_sift.txt` or `src/hyperparameter_tuning_logs_siftsmall.txt`. csv dataframes will be found at `src/df_metrics.csv` or `src/df_metrics_siftsmall.csv`.
