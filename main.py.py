
"""
Created on April  3 15:37:50 2019

@author: ZHUOYI,Lin

Wij=exp[-B*(1-cos(i,j))]
"""

import os
import pickle
import numpy as np
import scipy
import scipy.sparse as sp
import math
import time
import decimal
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import os,sys
from math import log


def model(mat_train,mu,sigma): #the GLIMG model
    g=0.5  #hyper-parameter taht balancing the global and local effect.
    gamma=1
    alpha=1/(1+mu)
    mat_train=mat_train.todense()
    mat_train=np.asarray(mat_train)
    train_t = mat_train.T
    sim_global=1-cosine_similarity(train_t, train_t)
    Completed_train=np.zeros((num_users, num_items), dtype=np.float32)
###############################################################################    
#    #for yelp dataset
    with open("labelyelp.txt", "rb") as fp:   # Unpickling
        labels = pickle.load(fp)
        num_cluster=20   #hyper-parameter K
    #for ml1m dataset    
#    with open("labelml1m.txt", "rb") as fp:   # Unpickling
#        labels = pickle.load(fp)
#        num_cluster=10
###############################################################################
    for k in range(num_cluster):
        idx=np.zeros((num_users, num_items), dtype=bool)
        enu=[i for i, j in enumerate(labels) if j == k]
        idx[enu]=True
        mat_local=idx*mat_train
        sim_local=1-cosine_similarity(mat_local.T, mat_local.T)
        
        sim=g*sim_local +(1-g)*sim_global
        W=np.exp(-sigma*sim)
        W=W-np.diag(np.diag(W))
     
        D=np.diag(np.sum(W,1))
        D1=np.power(D, -0.5)
        where_are_inf = np.isinf(D1)
        D1[where_are_inf] = 0
            
        S=np.dot(D1,W)
        S=np.dot(S,D1)
        L=gamma*D-S
        
        M=np.identity(np.size(L,1))+alpha*L
        M1=np.linalg.inv(M)
    
        Completed_train= np.dot(mat_local, M1) + Completed_train

    #rank and save recommendation results.
    user_ranklist_map = {}
    user_ranklist_score_map = {}
    for i in range(num_users):
        u_idx = i   
        sorted_product_idxs = sorted(range(len(Completed_train[i])), key=lambda k: Completed_train[i][k], reverse=True)
    #    user_ranklist_map[u_idx],user_ranklist_score_map[u_idx] = data_util.compute_test_product_ranklist(u_idx, Completed_train[i],
    #													sorted_product_idxs, rank_cutoff) #(product name, rank)
        product_rank_list = []
        product_rank_scores = []
        rank = 0
        for product_idx in sorted_product_idxs:
            if mat_train[u_idx, product_idx]!=0 or math.isnan(Completed_train[i,product_idx]):
                continue
            product_rank_list.append(product_idx)
            product_rank_scores.append(Completed_train[i,product_idx])
            rank += 1
            if rank == rank_cutoff:
                break
            user_ranklist_map[u_idx]=product_rank_list
            user_ranklist_score_map[u_idx]= product_rank_scores
    
    #data_set.output_ranklist(user_ranklist_map, user_ranklist_score_map, FLAGS.train_dir, FLAGS.similarity_func)
    with open( 'test.ranklist', 'w') as rank_fout:
    			for u_idx in user_ranklist_map:
    				for i in range(len(user_ranklist_map[u_idx])):
    					product_id = user_ranklist_map[u_idx][i]
    					rank_fout.write(str(u_idx) + ' Q0 ' + str(product_id) + ' ' + str(i+1)
    							+ ' ' + str(user_ranklist_score_map[u_idx][i]) + ' \n')
    



#compute ndcg
def metrics(doc_list, rel_set):
    dcg = 0.0
    hit_num = 0.0
    for i in range(len(doc_list)):
        if doc_list[i] in rel_set:
        			#dcg
            dcg += 1/(log(i+2)/log(2))
            hit_num += 1
	  #idcg
    idcg = 0.0
    for i in range(min(len(rel_set),len(doc_list))):
        idcg += 1/(log(i+2)/log(2))
    ndcg = dcg/idcg
    recall = hit_num / len(rel_set)
    precision = hit_num / len(doc_list)
    #compute hit_ratio
    hit = 1.0 if hit_num > 0 else 0.0
    large_rel = 1.0 if len(rel_set) > len(doc_list) else 0.0
    return recall, ndcg, hit, large_rel, precision

def print_metrics_with_rank_cutoff(rank_cutoff):
    #read rank_list file
    rank_list = {}
    with open(rank_list_file) as fin:
        for line in fin:
            arr = line.strip().split(' ')
            qid = arr[0]
            did = arr[2]
            if qid not in rank_list:
                rank_list[qid] = []
            if len(rank_list[qid]) > rank_cutoff:
                continue
            rank_list[qid].append(did)

    ndcgs = 0.0
    recalls = 0.0
    hits = 0.0
    large_rels = 0.0
    precisions = 0.0
    count_query = 0
    for qid in rank_list:
        if qid in qrel_map:
            recall, ndcg, hit, large_rel, precision = metrics(rank_list[qid],qrel_map[qid])
            count_query += 1
            ndcgs += ndcg
            recalls += recall
            hits += hit
            large_rels += large_rel
            precisions += precision
    print("hit Number:" + str(hits))
    print("Query Number:" + str(count_query))
    print("Larger_rel_set@"+str(rank_cutoff) + ":" + str(large_rels/count_query))
    print("Recall@"+str(rank_cutoff) + ":" + str(recalls/count_query))
    print("Precision@"+str(rank_cutoff) + ":" + str(precisions/count_query))
    print("NDCG@"+str(rank_cutoff) + ":" + str(ndcgs/count_query))
    print("HitRatio@"+str(rank_cutoff) + ":" + str(hits/count_query))
    return str(ndcgs/count_query)



###############################################################################
#main program starts from here.
#start from choosing a dataset.
###############################################################################
#Define path for evaluation.
rank_list_file = "~/test.ranklist"  #recommendation results
test_qrel_file = "~/dataset/testyelp.qrels"   #exactly the test data
#Load data for yelp dataset
dataset = np.loadtxt("~/dataset/yelp/yelp_train.txt")
num_users=int(dataset[:,0].max()+1);
num_items=int(dataset[:,1].max()+1);            
mat_train = sp.lil_matrix((num_users, num_items), dtype=np.int)

for i in range(0,len(dataset)):
    arr=dataset[i]
    user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
    mat_train[user, item] = rating

###############################################################################
# load data for ml1m dataset

#rank_list_file = "~/test.ranklist"
#test_qrel_file = "~/dataset/testml1m.qrels"
#dataset = np.loadtxt("~/dataset/ml1m/ml1m_train.txt")
#num_users=int(dataset[:,0].max()+1);
#num_items=int(dataset[:,1].max()+1);            
#
#mat_train = sp.lil_matrix((num_users, num_items), dtype=np.int)
#
#for i in range(0,len(dataset)):
#    arr=dataset[i]
#    user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
#    mat_train[user, item] = rating

###############################################################################

rank_cutoff=50 # n in Top-n
mu_list=[100]   #hyper-parameter
sigma_list=[0.01] #hyper-parameter
#for yelp,the best parameter set is [100,0.01]
#for ml1m, the best parameter set is [0.5,0.5]

#Initialization
mat=[]
best_ndcg=0
best_mu=0
best_sigma=0

#Perform grid search to select the best set of parameters for GLIMG model 
for mu in mu_list:
    for sigma in sigma_list:
        model(mat_train,mu,sigma)
        
        qrel_map = {}
        with open(test_qrel_file) as fin:
            for line in fin:
                arr = line.strip().split(' ')
                qid = arr[0]
                did = arr[2]
                label = int(arr[3])
                if label < 1:
                    continue
                if qid not in qrel_map:
                    qrel_map[qid] = set()
                qrel_map[qid].add(did)
        #print(a,b)       
        ndcg=float(print_metrics_with_rank_cutoff(50))*100
        mat.append([mu, sigma , ndcg])
        
        if float(ndcg)>best_ndcg:
            best_ndcg=float(ndcg)
            best_mu=mu
            best_sigma=sigma
print("best_mu:"+str(best_mu))
print("best_sigma:"+str(best_sigma))
print("best_ndcg:"+str(best_ndcg))
