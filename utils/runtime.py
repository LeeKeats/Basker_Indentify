from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Iterable, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PATH_FIELDS: Tuple[Tuple[str, ...], ...] = (
    ("MODEL", "PRETRAIN_PATH"),
    ("DATASETS", "ROOT_DIR"),
    ("TEST", "WEIGHT"),
    ("OUTPUT_DIR",),
)


def build_argument_parser(description: str, include_local_rank: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if include_local_rank:
        parser.add_argument("--local_rank", default=0, type=int)
    return parser


def load_config_from_args(args: argparse.Namespace):
    from config import cfg as default_cfg

    cfg = default_cfg.clone()
    config_file = ""
    if args.config_file:
        config_file = resolve_path(args.config_file, must_exist=True)
        cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    normalize_config_paths(cfg, config_file=config_file)
    cfg.freeze()
    return cfg, config_file


def resolve_path(raw_path: str, config_file: str = "", must_exist: bool = False) -> str:
    if not raw_path:
        return ""

    path = Path(raw_path)
    if path.is_absolute():
        resolved = path.resolve()
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {resolved}")
        return str(resolved)

    config_dir = Path(config_file).resolve().parent if config_file else None
    candidates = [Path.cwd() / path]

    if PROJECT_ROOT != Path.cwd():
        candidates.append(PROJECT_ROOT / path)
    if config_dir is not None:
        candidates.append(config_dir / path)

    resolved_candidates = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        resolved_candidates.append(resolved)

    if must_exist:
        for candidate in resolved_candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(f"Path does not exist: {raw_path}")

    return str(resolved_candidates[0])


def normalize_config_paths(cfg, config_file: str = "") -> None:
    cfg.defrost()
    for field_path in _PATH_FIELDS:
        current_value = _get_cfg_value(cfg, field_path)
        if not current_value:
            continue
        _set_cfg_value(cfg, field_path, resolve_path(current_value, config_file=config_file))


def configure_runtime(cfg, local_rank: int = 0) -> None:
    import torch

    if str(cfg.MODEL.DEVICE).lower() == "cuda":
        device_ids = normalize_device_ids(cfg.MODEL.DEVICE_ID)
        if device_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
        if cfg.MODEL.DIST_TRAIN and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)


def ensure_output_dir(cfg) -> str:
    output_dir = str(cfg.OUTPUT_DIR)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def validate_run_config(cfg, require_test_weight: bool = False) -> None:
    dataset_root = Path(str(cfg.DATASETS.ROOT_DIR))
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    if require_test_weight:
        weight_path = Path(str(cfg.TEST.WEIGHT))
        if not weight_path.exists():
            raise FileNotFoundError(f"Test weight does not exist: {weight_path}")


def log_run_setup(logger: logging.Logger, args: argparse.Namespace, cfg, config_file: str = "") -> None:
    logger.info(args)
    if config_file:
        logger.info("Loaded configuration file %s", config_file)
        with open(config_file, "r", encoding="utf-8", errors="replace") as config_stream:
            logger.info("\n%s", config_stream.read())
    logger.info("Running with config:\n%s", cfg)


def set_seed(seed: int, deterministic: bool = False) -> None:
    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def normalize_device_ids(raw_value) -> str:
    value = str(raw_value).strip()
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1]
    return value.replace("'", "").replace('"', "").replace(" ", "")


def _get_cfg_value(cfg, field_path: Iterable[str]):
    node = cfg
    for field_name in field_path:
        node = getattr(node, field_name)
    return node


def _set_cfg_value(cfg, field_path: Iterable[str], value) -> None:
    node = cfg
    field_path = list(field_path)
    for field_name in field_path[:-1]:
        node = getattr(node, field_name)
    setattr(node, field_path[-1], value)
