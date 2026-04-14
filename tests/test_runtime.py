import unittest

from utils.runtime import normalize_device_ids, resolve_path


class RuntimeTests(unittest.TestCase):
    def test_normalize_device_ids_strips_wrapping_tuple_and_quotes(self):
        self.assertEqual(normalize_device_ids("('0, 1')"), "0,1")

    def test_resolve_path_handles_existing_relative_file(self):
        resolved = resolve_path("README.md", must_exist=True)
        self.assertTrue(resolved.endswith("README.md"))


if __name__ == "__main__":
    unittest.main()
