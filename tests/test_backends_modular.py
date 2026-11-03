# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Dynamic Hardware Backend Modularization & Plugin Registry."""

import math
import unittest
import torch

from hydragen.backends import (
    BackendRegistry,
    CPUHardwareBackend,
    CPUPagedAttentionBackend,
    CUDAHardwareBackend,
    CUDAPagedAttentionBackend,
    DeviceCapability,
    DeviceType,
    FallbackIPCBackend,
    HardwareBackend,
    IPCBackend,
    MoEBackend,
    NativeIPCBackend,
    PagedAttentionBackend,
    get_hardware_backend,
    get_ipc_backend,
    get_paged_attention_backend,
)
from hydragen.paged_kv import PagedKVManager


class TestHardwareBackendsModular(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_device_capability_cpu(self):
        """Validates CPU device capability introspection."""
        backend = CPUHardwareBackend()
        cap = backend.capability
        self.assertEqual(cap.device_type, DeviceType.CPU)
        self.assertTrue(len(cap.device_name) > 0)
        self.assertTrue(cap.multi_processor_count >= 1)

    def test_device_capability_cuda(self):
        """Validates CUDA device capability introspection if CUDA is available."""
        backend = CUDAHardwareBackend()
        cap = backend.capability
        self.assertEqual(cap.device_type, DeviceType.CUDA)
        if torch.cuda.is_available():
            self.assertTrue(cap.compute_capability[0] > 0)
            self.assertTrue(cap.total_memory_bytes > 0)
            self.assertIn("is_ampere", cap.details)

    def test_backend_registry_factory(self):
        """Validates default backend resolution through BackendRegistry."""
        hw = get_hardware_backend()
        self.assertIsInstance(hw, HardwareBackend)

        paged_att = get_paged_attention_backend()
        self.assertIsInstance(paged_att, PagedAttentionBackend)
        self.assertTrue(paged_att.is_available())

        ipc = get_ipc_backend()
        self.assertIsInstance(ipc, IPCBackend)
        self.assertTrue(ipc.is_available())

    def test_forced_cpu_backend_selection(self):
        """Validates forcing CPU execution backend even on GPU machines."""
        paged_att = get_paged_attention_backend(force_device=DeviceType.CPU)
        self.assertIsInstance(paged_att, CPUPagedAttentionBackend)
        self.assertEqual(paged_att.device_type, DeviceType.CPU)
        self.assertEqual(paged_att.name, "cpu_paged_attention")

    def test_paged_kv_manager_backend_injection(self):
        """Verifies PagedKVManager accepts explicit backend injection for modular extensibility."""
        custom_backend = CPUPagedAttentionBackend()
        mgr = PagedKVManager(
            block_size=16,
            num_blocks=16,
            num_layers=1,
            num_kv_heads=2,
            head_dim=64,
            backend=custom_backend,
        )
        self.assertIs(mgr.backend, custom_backend)

        # Allocate sequence and test attention through injected backend
        seq_id = 99
        mgr.allocate_sequence(seq_id, 8)
        bl = mgr.get_block_table(seq_id)
        block_tables = torch.tensor([bl + [0] * (4 - len(bl))], dtype=torch.int32)
        context_lens = torch.tensor([8], dtype=torch.int32)

        q = torch.randn(1, 4, 64)
        k_cache = torch.randn(16, 2, 16, 64)
        v_cache = torch.randn(16, 2, 16, 64)
        out = torch.zeros_like(q)

        mgr.paged_attention_decode(
            out=out,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
        )
        self.assertFalse(torch.isnan(out).any())

    def test_fallback_ipc_backend(self):
        """Validates queue-based FallbackIPCBackend."""
        ipc = FallbackIPCBackend()
        self.assertTrue(ipc.is_available())
        handle = ipc.create(stdout_fd=1, num_slots=2)
        self.assertIsNotNone(handle)

        # Test token read timeout
        tok = ipc.read_token(handle, slot=0, timeout_ms=10)
        self.assertIsNone(tok)

        # Enqueue token directly and read
        handle["slots"][0].put(42)
        tok = ipc.read_token(handle, slot=0, timeout_ms=100)
        self.assertEqual(tok, 42)

        # Clear slot
        handle["slots"][1].put(100)
        ipc.clear_slot(handle, slot=1)
        self.assertIsNone(ipc.read_token(handle, slot=1, timeout_ms=10))

    def test_custom_hardware_plugin_extensibility(self):
        """Verifies that 3rd-party hardware backends can implement the interface seamlessly."""
        class MockNPUAttentionBackend(PagedAttentionBackend):
            @property
            def name(self) -> str:
                return "mock_npu"

            @property
            def device_type(self) -> DeviceType:
                return DeviceType.UNKNOWN

            def is_available(self) -> bool:
                return True

            def paged_attention_decode(self, out, q, *args, **kwargs):
                out.fill_(7.0)
                return out

            def paged_attention_prefill(self, out, q, *args, **kwargs):
                out.fill_(3.0)
                return out

        npu_att = MockNPUAttentionBackend()
        q = torch.zeros(1, 2, 16)
        out = torch.zeros_like(q)
        npu_att.paged_attention_decode(out, q, None, None, None, None)
        self.assertTrue((out == 7.0).all())

        npu_att.paged_attention_prefill(out, q, None, None, None)
        self.assertTrue((out == 3.0).all())


if __name__ == "__main__":
    unittest.main()
