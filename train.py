import torch

from datasets import make_dataloader
from loss import make_loss
from model import make_model
from processor import do_train
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from utils.logger import setup_logger
from utils.runtime import (
    build_argument_parser,
    configure_runtime,
    ensure_output_dir,
    load_config_from_args,
    log_run_setup,
    set_seed,
    validate_run_config,
)

if __name__ == '__main__':
    parser = build_argument_parser("ReID Baseline Training", include_local_rank=True)
    args = parser.parse_args()

    cfg, config_file = load_config_from_args(args)
    validate_run_config(cfg)
    configure_runtime(cfg, local_rank=args.local_rank)

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN and torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    output_dir = ensure_output_dir(cfg)
    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :%s", cfg.OUTPUT_DIR)
    log_run_setup(logger, args, cfg, config_file)

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )
