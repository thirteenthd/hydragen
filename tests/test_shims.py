# SPDX-License-Identifier: Apache-2.0
"""Test module forwarding shims between serve and hydragen."""
import os
import sys
import unittest

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


class TestModuleShims(unittest.TestCase):
    def test_shims_module_identity(self):
        import serve.server
        import hydragen.server.server
        self.assertIs(serve.server, hydragen.server.server)

        import serve.frontend
        import hydragen.server.frontend
        self.assertIs(serve.frontend, hydragen.server.frontend)

        import serve.trtllm_engine
        import hydragen.trtllm_engine
        self.assertIs(serve.trtllm_engine, hydragen.trtllm_engine)

        import serve.fast_ipc
        import hydragen.fast_ipc
        self.assertIs(serve.fast_ipc, hydragen.fast_ipc)

        import serve.telemetry
        import hydragen.server.telemetry
        self.assertIs(serve.telemetry, hydragen.server.telemetry)


if __name__ == "__main__":
    unittest.main()
