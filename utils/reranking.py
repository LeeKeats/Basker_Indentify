#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri, 25 May 2018 20:29:09


"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API

probFea: all feature vectors of the query set (torch tensor)
probFea: all feature vectors of the gallery set (torch tensor)
k1,k2,lambda: parameters, the original paper is (k1=20,k2=6,lambda=0.3)
MemorySave: set to 'True' when using MemorySave mode
Minibatch: avaliable when 'MemorySave' is 'True'
"""

import numpy as np


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """
    q_g_dist: [num_query, num_gallery]
    q_q_dist: [num_query, num_query]
    g_g_dist: [num_gallery, num_gallery]
    return:   [num_query, num_gallery]
    """

    original_dist = np.concatenate(
        [
            np.concatenate([q_q_dist, q_g_dist], axis=1),
            np.concatenate([q_g_dist.T, g_g_dist], axis=1),
        ],
        axis=0,
    ).astype(np.float32)

    # 标准实现里会再做一次整体归一化
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = original_dist / np.max(original_dist, axis=0, keepdims=True)
    original_dist = np.transpose(original_dist)

    all_num = original_dist.shape[0]
    query_num = q_g_dist.shape[0]

    V = np.zeros_like(original_dist, dtype=np.float32)
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)

    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]

        k_reciprocal_expansion_index = k_reciprocal_index.copy()
        for candidate in k_reciprocal_index:
            candidate_forward_k_neigh_index = initial_rank[candidate, : int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, : int(np.around(k1 / 2)) + 1
            ]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]

            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > (
                2.0 / 3.0 * len(candidate_k_reciprocal_index)
            ):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe

    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros((1, all_num), dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]

        for j, ind in enumerate(indNonZero):
            temp_min[0, indImages[j]] += np.minimum(V[i, ind], V[indImages[j], ind])

        jaccard_dist[i] = 1.0 - temp_min / (2.0 - temp_min)

    # final_dist = lambda * original + (1-lambda) * jaccard
    final_dist = jaccard_dist[:, query_num:] * (1.0 - lambda_value) + \
                 original_dist[:query_num, query_num:] * lambda_value

    return final_dist