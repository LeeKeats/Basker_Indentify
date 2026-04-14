import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from config import cfg
from datasets import make_dataloader
from model.backbones.resnet_ibn import ResNet50IBNReID


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
            feat = model(img)

            feats.append(feat.cpu())
            pids.extend(to_list(pid))
            camids.extend(to_list(camid))

    feats = torch.cat(feats, dim=0)
    pids = np.asarray(pids)
    camids = np.asarray(camids)
    return feats, pids, camids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ResNet50-IBN Features")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.defrost()
    cfg.DATASETS.NAMES = 'ballshow'
    cfg.DATASETS.ROOT_DIR = 'F:/TransReID-master/data'
    cfg.freeze()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = ResNet50IBNReID(num_classes=num_classes).to(device)

    checkpoint = torch.load(cfg.TEST.WEIGHT, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=False)

    feats, pids, camids = extract_features(model, val_loader, device)

    qf = feats[:num_query].numpy()
    gf = feats[num_query:].numpy()
    q_pids = pids[:num_query]
    g_pids = pids[num_query:]
    q_camids = camids[:num_query]
    g_camids = camids[num_query:]

    save_dir = "features/resnet50_ibn_v2"
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "qf.npy"), qf)
    np.save(os.path.join(save_dir, "gf.npy"), gf)
    np.save(os.path.join(save_dir, "q_pids.npy"), q_pids)
    np.save(os.path.join(save_dir, "g_pids.npy"), g_pids)
    np.save(os.path.join(save_dir, "q_camids.npy"), q_camids)
    np.save(os.path.join(save_dir, "g_camids.npy"), g_camids)

    print("ResNet50 features saved successfully.")