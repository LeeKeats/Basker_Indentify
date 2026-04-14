import glob
import os.path as osp
import re

from .bases import BaseImageDataset


class BallShow(BaseImageDataset):
    """BallShow person re-identification dataset."""

    dataset_dir = "BallShow"
    _FILENAME_PATTERN = re.compile(r"([-\d]+)_c(\d+)")

    def __init__(self, root="", verbose=True, pid_begin=0, **kwargs):
        super(BallShow, self).__init__()
        self.dataset_dir = self._resolve_dataset_dir(root)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")

        self._check_before_run()
        self.pid_begin = pid_begin

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print(f"=> BallShow loaded from {self.dataset_dir}")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train
        )
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query
        )
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = (
            self.get_imagedata_info(self.gallery)
        )

    @classmethod
    def _resolve_dataset_dir(cls, root):
        normalized_root = osp.normpath(root) if root else ""
        if normalized_root and osp.basename(normalized_root).lower() == cls.dataset_dir.lower():
            return normalized_root
        return osp.join(normalized_root, cls.dataset_dir)

    @classmethod
    def _parse_filename(cls, img_path):
        match = cls._FILENAME_PATTERN.search(osp.basename(img_path))
        if match is None:
            raise ValueError("Unexpected BallShow filename format: '{}'".format(img_path))
        pid, camid = map(int, match.groups())
        return pid, camid

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = self._parse_filename(img_path)
            if pid == -1:
                continue
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = self._parse_filename(img_path)

            if pid == -1:
                continue

            camid -= 1
            if relabel:
                pid = pid2label[pid]
            else:
                pid = self.pid_begin + pid

            dataset.append((img_path, pid, camid, 1))

        return dataset
