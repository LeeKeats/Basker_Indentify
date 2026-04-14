import torch
import torch.distributed as dist
import torchvision.transforms as T
from timm.data.random_erasing import RandomErasing
from torch.utils.data import DataLoader

from .ballshow import BallShow
from .bases import ImageDataset
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP

__factory = {
    "ballshow": BallShow,
}


def train_collate_fn(batch):
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg):
    if cfg.DATASETS.NAMES not in __factory:
        supported_datasets = ", ".join(sorted(__factory.keys()))
        raise KeyError(f"Unsupported dataset '{cfg.DATASETS.NAMES}'. Available datasets: {supported_datasets}")

    train_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"),
        ]
    )

    val_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )

    loader_kwargs = {
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if "triplet" in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print("DIST_TRAIN START")
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(
                dataset.train,
                cfg.SOLVER.IMS_PER_BATCH,
                cfg.DATALOADER.NUM_INSTANCE,
            )
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = DataLoader(
                train_set,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                **loader_kwargs,
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(
                    dataset.train,
                    cfg.SOLVER.IMS_PER_BATCH,
                    cfg.DATALOADER.NUM_INSTANCE,
                ),
                collate_fn=train_collate_fn,
                **loader_kwargs,
            )
    elif cfg.DATALOADER.SAMPLER == "softmax":
        print("using softmax sampler")
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            shuffle=True,
            collate_fn=train_collate_fn,
            **loader_kwargs,
        )
    else:
        raise ValueError(
            "Unsupported sampler '{}', expected softmax or a sampler containing triplet".format(
                cfg.DATALOADER.SAMPLER
            )
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        collate_fn=val_collate_fn,
        **loader_kwargs,
    )
    train_loader_normal = DataLoader(
        train_set_normal,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        collate_fn=val_collate_fn,
        **loader_kwargs,
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
