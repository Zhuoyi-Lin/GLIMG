# GLIMG
This is our implementation for the paper:

Zhuoyi Lin, Lei Feng, Rui Yin, Chi Xu, Chee-Keong Kwoh, "GLIMG: Global and Local Item Graphs for Top-N Recommender Systems", Information Sciences.

Please cite our Informqation Science paper if you use our codes. Thanks!

# Dataset

We provide two processed datasets: MovieLens 1 Million (ml-1m) and Yelp (Yelp Challenge Dataset on January 2018). We split each dataset into 3 parts, the first 80%
are used for training, 10% are used for validation and the remaining 10 % are used for testing the performance.
In addition, Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG) are the eavaluation metrics in this work.

# Cluster Labels for 2 datasets
We provide the cluster labels for each user using K-means++, which is shown in the "label.txt" documents.

# Hyper-paramters
There are four hyper-paramters(k: number of clusters; g: balancing global and local effect; \mu and sigma), whose ranges are presented in our paper. We state the best set of parameters in our paper and code.
