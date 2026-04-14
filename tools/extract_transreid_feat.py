import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from config import cfg
from datasets import make_dataloader
from model import make_model


def to_list(x):
    if torch.is_tensor(x):
        return x.cpu().numpy().tolist()
    return list(x)


def extract_features(model, val_loader, device):
    model.eval()
    feats = []
    pids = []
    camids = []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            img, pid, camid, camids_batch, viewid, imgpath = batch

            img = img.to(device)
            camids_batch = camids_batch.to(device)
            viewid = viewid.to(device)

            feat = model(img, cam_label=camids_batch, view_label=viewid)

            feats.append(feat.cpu())
            pids.extend(to_list(pid))
            camids.extend(to_list(camid))

    feats = torch.cat(feats, dim=0)
    pids = np.asarray(pids)
    camids = np.asarray(camids)
    return feats, pids, camids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract TransReID Features")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num
    )
    model = model.to(device)

    model.load_param(cfg.TEST.WEIGHT)

    feats, pids, camids = extract_features(model, val_loader, device)

    qf = feats[:num_query].numpy()
    gf = feats[num_query:].numpy()
    q_pids = pids[:num_query]
    g_pids = pids[num_query:]
    q_camids = camids[:num_query]
    g_camids = camids[num_query:]

    os.makedirs("features/transreid", exist_ok=True)
    np.save("features/transreid/qf.npy", qf)
    np.save("features/transreid/gf.npy", gf)
    np.save("features/transreid/q_pids.npy", q_pids)
    np.save("features/transreid/g_pids.npy", g_pids)
    np.save("features/transreid/q_camids.npy", q_camids)
    np.save("features/transreid/g_camids.npy", g_camids)

    print("TransReID features saved successfully.")