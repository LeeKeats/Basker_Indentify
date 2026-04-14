import os
import numpy as np
from utils.metrics import eval_func
from utils.reranking import re_ranking


def l2_normalize(feat, axis=1, eps=1e-12):
    norm = np.linalg.norm(feat, ord=2, axis=axis, keepdims=True)
    return feat / (norm + eps)


def euclidean_distmat(x, y):
    x2 = np.sum(np.square(x), axis=1, keepdims=True)
    y2 = np.sum(np.square(y), axis=1, keepdims=True).T
    dist = x2 + y2 - 2 * np.matmul(x, y.T)
    dist = np.maximum(dist, 0.0)
    return dist.astype(np.float32)


def load_feat_dir(feat_dir):
    qf = np.load(os.path.join(feat_dir, "qf.npy"))
    gf = np.load(os.path.join(feat_dir, "gf.npy"))
    q_pids = np.load(os.path.join(feat_dir, "q_pids.npy"))
    g_pids = np.load(os.path.join(feat_dir, "g_pids.npy"))
    q_camids = np.load(os.path.join(feat_dir, "q_camids.npy"))
    g_camids = np.load(os.path.join(feat_dir, "g_camids.npy"))
    return qf, gf, q_pids, g_pids, q_camids, g_camids


if __name__ == "__main__":
    vit_dir = "features/transreid"
    cnn_dir = "features/resnet50_ibn_v2"

    
    alpha = 0.80

    k1 = 10
    k2 = 3
    lambda_value = 0.7

    qf_vit, gf_vit, q_pids_vit, g_pids_vit, q_camids_vit, g_camids_vit = load_feat_dir(vit_dir)
    qf_cnn, gf_cnn, q_pids_cnn, g_pids_cnn, q_camids_cnn, g_camids_cnn = load_feat_dir(cnn_dir)

    assert np.array_equal(q_pids_vit, q_pids_cnn), "q_pids not aligned!"
    assert np.array_equal(g_pids_vit, g_pids_cnn), "g_pids not aligned!"

    q_pids = q_pids_vit
    g_pids = g_pids_vit
    q_camids = q_camids_vit
    g_camids = g_camids_vit

    qf_vit = l2_normalize(qf_vit)
    gf_vit = l2_normalize(gf_vit)
    qf_cnn = l2_normalize(qf_cnn)
    gf_cnn = l2_normalize(gf_cnn)

    q_g_vit = euclidean_distmat(qf_vit, gf_vit)
    q_q_vit = euclidean_distmat(qf_vit, qf_vit)
    g_g_vit = euclidean_distmat(gf_vit, gf_vit)

    q_g_cnn = euclidean_distmat(qf_cnn, gf_cnn)
    q_q_cnn = euclidean_distmat(qf_cnn, qf_cnn)
    g_g_cnn = euclidean_distmat(gf_cnn, gf_cnn)

    q_g_fuse = alpha * q_g_vit + (1.0 - alpha) * q_g_cnn
    q_q_fuse = alpha * q_q_vit + (1.0 - alpha) * q_q_cnn
    g_g_fuse = alpha * g_g_vit + (1.0 - alpha) * g_g_cnn

    rerank_dist = re_ranking(
        q_g_fuse,
        q_q_fuse,
        g_g_fuse,
        k1=k1,
        k2=k2,
        lambda_value=lambda_value
    )

    cmc, mAP = eval_func(rerank_dist, q_pids, g_pids, q_camids, g_camids)

    print("========== Result ==========")
    print(f"Rank-1 : {cmc[0]:.4f}")
    print(f"Rank-5 : {cmc[4]:.4f}")
    print(f"Rank-10: {cmc[9]:.4f}")
    print(f"mAP    : {mAP:.4f}")