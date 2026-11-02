# SPDX-License-Identifier: Apache-2.0
"""Test suite for native library loader paths and resolution."""

import os
import sys
import unittest

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from hydragen.lib_loader import get_search_paths, load_native_library


class TestLibLoader(unittest.TestCase):
    def test_search_paths_include_canonical_locations(self):
        paths = [str(p) for p in get_search_paths("test_lib.so")]
        self.assertTrue(any("hydragen" in p and "lib" in p for p in paths))
        self.assertTrue(any("server" in p for p in paths))

    def test_search_paths_honor_env_var(self):
        from pathlib import Path
        custom_path = str(Path("/tmp/custom_lib.so"))
        os.environ["TEST_CUSTOM_LIB_SO"] = custom_path
        paths = [str(p) for p in get_search_paths("test_lib.so", env_var="TEST_CUSTOM_LIB_SO")]
        self.assertEqual(paths[0], custom_path)
        del os.environ["TEST_CUSTOM_LIB_SO"]

    def test_load_nonexistent_library_returns_none(self):
        lib = load_native_library("nonexistent_test_lib_12345.so")
        self.assertIsNone(lib)

    def test_paged_kv_native_load(self):
        from hydragen.paged_kv import PagedKVManager
        mgr = PagedKVManager(block_size=16, num_blocks=128)
        # If running on Linux with built shared lib, handle should be active
        if sys.platform.startswith("linux"):
            self.assertIsNotNone(mgr._handle)



if __name__ == "__main__":
    unittest.main()
