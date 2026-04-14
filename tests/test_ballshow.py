import unittest

from datasets.ballshow import BallShow


class BallShowTests(unittest.TestCase):
    def test_parse_filename_supports_multi_digit_camera_id(self):
        pid, camid = BallShow._parse_filename("0015503_c0795s1312_000003_03.jpg")
        self.assertEqual(pid, 15503)
        self.assertEqual(camid, 795)

    def test_resolve_dataset_dir_accepts_project_root_data_path(self):
        dataset_dir = BallShow._resolve_dataset_dir("data")
        self.assertEqual(dataset_dir.replace("\\", "/"), "data/BallShow")

    def test_resolve_dataset_dir_avoids_double_ballshow_suffix(self):
        dataset_dir = BallShow._resolve_dataset_dir("data/BallShow")
        self.assertEqual(dataset_dir.replace("\\", "/"), "data/BallShow")


if __name__ == "__main__":
    unittest.main()
