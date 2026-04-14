import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda import amp

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval


def _resolve_device(cfg, local_rank=0):
    if str(cfg.MODEL.DEVICE).lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if cfg.MODEL.DIST_TRAIN:
        return torch.device("cuda", local_rank)
    return torch.device("cuda")


def _run_validation(model, val_loader, evaluator, device):
    model.eval()
    for _, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, vid, camid))
    return evaluator.compute()


def do_train(
    cfg,
    model,
    center_criterion,
    train_loader,
    val_loader,
    optimizer,
    optimizer_center,
    scheduler,
    loss_fn,
    num_query,
    local_rank,
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    device = _resolve_device(cfg, local_rank=local_rank)
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info("start training")

    model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        print("Using {} GPUs for training".format(torch.cuda.device_count()))
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
        )

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        model.train()

        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=device.type == "cuda"):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if "center" in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= 1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT
                scaler.step(optimizer_center)
                scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc.item(), 1)

            if device.type == "cuda":
                torch.cuda.synchronize()

            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                        epoch,
                        n_iter + 1,
                        len(train_loader),
                        loss_meter.avg,
                        acc_meter.avg,
                        scheduler._get_lr(epoch)[0],
                    )
                )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if not cfg.MODEL.DIST_TRAIN:
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch,
                    time_per_batch,
                    train_loader.batch_size / time_per_batch,
                )
            )

        if epoch % checkpoint_period == 0:
            checkpoint_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch))
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(), checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    evaluator.reset()
                    cmc, mAP, _, _, _, _, _ = _run_validation(model, val_loader, evaluator, device)
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            else:
                evaluator.reset()
                cmc, mAP, _, _, _, _, _ = _run_validation(model, val_loader, evaluator, device)
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                if device.type == "cuda":
                    torch.cuda.empty_cache()


def do_inference(cfg, model, val_loader, num_query):
    device = _resolve_device(cfg)
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print("Using {} GPUs for inference".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    model.eval()
    for _, (img, pid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]

