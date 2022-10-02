# [Information Sciences] GLIMG
This is our implementation for the paper:

Zhuoyi Lin, Lei Feng, Rui Yin, Chi Xu, Chee-Keong Kwoh, "GLIMG: Global and Local Item Graphs for Top-N Recommender Systems", Information Sciences.
[Link](https://www.sciencedirect.com/science/article/abs/pii/S0020025521008185)

Please cite our Information Science paper if you use our codes. Thanks!
```
@article{LIN20211,
title = {GLIMG: Global and local item graphs for top-N recommender systems},
journal = {Information Sciences},
volume = {580},
pages = {1-14},
year = {2021},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2021.08.018},
url = {https://www.sciencedirect.com/science/article/pii/S0020025521008185},
author = {Zhuoyi Lin and Lei Feng and Rui Yin and Chi Xu and Chee Keong Kwoh},
keywords = {Item graph, Local model, Top-N recommendation},
abstract = {Graph-based recommendation models work well for top-N recommender systems due to their capability to capture the potential relationships between entities. However, most of the existing methods only construct a single global item graph shared by all the users and regrettably ignore the diverse tastes between different user groups. Inspired by the success of local models for recommendation tasks, this paper provides the first attempt to investigate multiple local item graphs along with a global item graph for graph-based recommendation models. We argue that recommendation on global and local graphs outperforms that on a single global graph or multiple local graphs. Specifically, we propose a novel graph-based recommendation model named GLIMG (Global and Local IteM Graphs), which simultaneously captures both the global and local user tastes. By integrating the global and local graphs into an adapted semi-supervised learning model, users’ preferences on items are propagated globally and locally. Extensive experimental results on real-world datasets show that our proposed method consistently outperforms the state-of-the-art counterparts on the top-N recommendation task.}
}
```

# Datasets

We provide two processed datasets: MovieLens 1 Million (ml-1m) and Yelp (Yelp Challenge Dataset on January 2018). We split each dataset into 3 parts, the first 80%
are used for training, 10% are used for validation and the remaining 10 % are used for testing the performance.
In addition, Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG) are the eavaluation metrics in this work.

# Cluster Labels for 2 datasets
We provide the cluster labels for each user using K-means++, which is shown in the "label.txt" documents.

# Hyper-paramters
There are four hyper-paramters(k: number of clusters; g: balancing global and local effect; \mu and sigma), whose ranges are presented in our paper. We state the best set of parameters in our paper and code.

# Contact
If you have any questions, please feel free to email me (ZHUOYI001@ntu.edu.sg).
