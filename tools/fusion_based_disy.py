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


def print_result(title, cmc, mAP):
    print(f"\n========== {title} ==========")
    print(f"Rank-1 : {cmc[0]:.4f}")
    print(f"Rank-5 : {cmc[4]:.4f}")
    print(f"Rank-10: {cmc[9]:.4f}")
    print(f"mAP    : {mAP:.4f}")


def search_rerank(dist_name, q_g_dist, q_q_dist, g_g_dist, q_pids, g_pids, q_camids, g_camids):
    best_r1 = -1.0
    best_map = -1.0
    best_cfg = None
    best_dist = None

    k1_list = [5, 10, 15]
    k2_list = [1, 3]
    lambda_list = [0.5, 0.7, 0.9]

    print(f"\n========== {dist_name} re-ranking search ==========")

    for k1 in k1_list:
        for k2 in k2_list:
            for lambda_value in lambda_list:
                rerank_dist = re_ranking(
                    q_g_dist,
                    q_q_dist,
                    g_g_dist,
                    k1=k1,
                    k2=k2,
                    lambda_value=lambda_value,
                )

                cmc, mAP = eval_func(rerank_dist, q_pids, g_pids, q_camids, g_camids)

                print(
                    f"{dist_name} | k1={k1:2d}, k2={k2:1d}, lambda={lambda_value:.1f} "
                    f"=> Rank-1: {cmc[0]:.4f}, mAP: {mAP:.4f}"
                )

                if (cmc[0] > best_r1) or (cmc[0] == best_r1 and mAP > best_map):
                    best_r1 = cmc[0]
                    best_map = mAP
                    best_cfg = (k1, k2, lambda_value)
                    best_dist = rerank_dist.copy()

    print("\n---------- Best re-ranking result ----------")
    print(f"{dist_name} best k1={best_cfg[0]}, k2={best_cfg[1]}, lambda={best_cfg[2]:.1f}")
    print(f"{dist_name} best Rank-1 = {best_r1:.4f}")
    print(f"{dist_name} best mAP    = {best_map:.4f}")

    return best_cfg, best_dist, best_r1, best_map


def search_three_weights(
    q_g_vit, q_q_vit, g_g_vit,
    q_g_cnn, q_q_cnn, g_g_cnn,
    q_g_os, q_q_os, g_g_os,
    q_pids, g_pids, q_camids, g_camids
):
    weight_candidates = [
        (0.75, 0.15, 0.10),
        (0.70, 0.15, 0.15),
        (0.70, 0.10, 0.20),
        (0.80, 0.10, 0.10),
        (0.75, 0.10, 0.15),
        (0.65, 0.20, 0.15),
    ]

    best_r1 = -1.0
    best_map = -1.0
    best_w = None
    best_qg = None
    best_qq = None
    best_gg = None

    print("\n========== Three-way fusion weight search ==========")

    for a, b, c in weight_candidates:
        q_g_fuse = a * q_g_vit + b * q_g_cnn + c * q_g_os
        q_q_fuse = a * q_q_vit + b * q_q_cnn + c * q_q_os
        g_g_fuse = a * g_g_vit + b * g_g_cnn + c * g_g_os

        cmc, mAP = eval_func(q_g_fuse, q_pids, g_pids, q_camids, g_camids)

        print(
            f"(vit,resnet,osnet)=({a:.2f},{b:.2f},{c:.2f}) "
            f"=> Rank-1: {cmc[0]:.4f}, mAP: {mAP:.4f}"
        )

        if (cmc[0] > best_r1) or (cmc[0] == best_r1 and mAP > best_map):
            best_r1 = cmc[0]
            best_map = mAP
            best_w = (a, b, c)
            best_qg = q_g_fuse.copy()
            best_qq = q_q_fuse.copy()
            best_gg = g_g_fuse.copy()

    print("\n---------- Best three-way fusion ----------")
    print(f"best weights = {best_w}")
    print(f"best Rank-1 = {best_r1:.4f}")
    print(f"best mAP    = {best_map:.4f}")

    return best_w, best_qg, best_qq, best_gg, best_r1, best_map


if __name__ == "__main__":
    vit_dir = "features/transreid"
    cnn_dir = "features/resnet50_ibn_v2"
    osnet_dir = "features/osnet_x0_75"   

    qf_vit, gf_vit, q_pids_vit, g_pids_vit, q_camids_vit, g_camids_vit = load_feat_dir(vit_dir)
    qf_cnn, gf_cnn, q_pids_cnn, g_pids_cnn, q_camids_cnn, g_camids_cnn = load_feat_dir(cnn_dir)
    qf_os, gf_os, q_pids_os, g_pids_os, q_camids_os, g_camids_os = load_feat_dir(osnet_dir)

    assert np.array_equal(q_pids_vit, q_pids_cnn), "q_pids vit/cnn not aligned!"
    assert np.array_equal(g_pids_vit, g_pids_cnn), "g_pids vit/cnn not aligned!"
    assert np.array_equal(q_pids_vit, q_pids_os), "q_pids vit/osnet not aligned!"
    assert np.array_equal(g_pids_vit, g_pids_os), "g_pids vit/osnet not aligned!"

    # 统一用 TransReID 的标签做评估
    q_pids = q_pids_vit
    g_pids = g_pids_vit
    q_camids = q_camids_vit
    g_camids = g_camids_vit

    # 特征 L2 normalize
    qf_vit = l2_normalize(qf_vit)
    gf_vit = l2_normalize(gf_vit)

    qf_cnn = l2_normalize(qf_cnn)
    gf_cnn = l2_normalize(gf_cnn)

    qf_os = l2_normalize(qf_os)
    gf_os = l2_normalize(gf_os)

    # 三路距离
    q_g_vit = euclidean_distmat(qf_vit, gf_vit)
    q_q_vit = euclidean_distmat(qf_vit, qf_vit)
    g_g_vit = euclidean_distmat(gf_vit, gf_vit)

    q_g_cnn = euclidean_distmat(qf_cnn, gf_cnn)
    q_q_cnn = euclidean_distmat(qf_cnn, qf_cnn)
    g_g_cnn = euclidean_distmat(gf_cnn, gf_cnn)

    q_g_os = euclidean_distmat(qf_os, gf_os)
    q_q_os = euclidean_distmat(qf_os, qf_os)
    g_g_os = euclidean_distmat(gf_os, gf_os)

    # 单模型 baseline
    cmc, mAP = eval_func(q_g_vit, q_pids, g_pids, q_camids, g_camids)
    print_result("TransReID only", cmc, mAP)
    vit_base_r1, vit_base_map = cmc[0], mAP

    cmc, mAP = eval_func(q_g_cnn, q_pids, g_pids, q_camids, g_camids)
    print_result("ResNet50 only", cmc, mAP)

    cmc, mAP = eval_func(q_g_os, q_pids, g_pids, q_camids, g_camids)
    print_result("OSNet only", cmc, mAP)

    # 三路权重搜索
    best_w, best_qg, best_qq, best_gg, fuse_base_r1, fuse_base_map = search_three_weights(
        q_g_vit, q_q_vit, g_g_vit,
        q_g_cnn, q_q_cnn, g_g_cnn,
        q_g_os, q_q_os, g_g_os,
        q_pids, g_pids, q_camids, g_camids
    )

    # 三路融合不加 rerank
    cmc, mAP = eval_func(best_qg, q_pids, g_pids, q_camids, g_camids)
    print_result(
        f"Three-way Fusion before re-ranking weights={best_w}",
        cmc, mAP
    )

    # 三路融合加 rerank
    fuse_best_cfg, fuse_best_dist, fuse_best_r1, fuse_best_map = search_rerank(
        "Three-way Fusion",
        best_qg, best_qq, best_gg,
        q_pids, g_pids, q_camids, g_camids
    )

    cmc, mAP = eval_func(fuse_best_dist, q_pids, g_pids, q_camids, g_camids)
    print_result(
        f"Three-way Fusion + re-ranking (k1={fuse_best_cfg[0]}, k2={fuse_best_cfg[1]}, lambda={fuse_best_cfg[2]:.1f})",
        cmc, mAP
    )

    print("\n========== Summary ==========")
    print(f"TransReID only                  : Rank-1 = {vit_base_r1:.4f}, mAP = {vit_base_map:.4f}")
    print(f"Three-way fusion best weights   : {best_w}")
    print(f"Three-way fusion before rerank  : Rank-1 = {fuse_base_r1:.4f}, mAP = {fuse_base_map:.4f}")
    print(f"Three-way fusion best rerank    : Rank-1 = {fuse_best_r1:.4f}, mAP = {fuse_best_map:.4f}")