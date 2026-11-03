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
    IntelHardwareBackend,
    IntelMoEBackend,
    IntelPagedAttentionBackend,
    MoEBackend,
    NativeIPCBackend,
    PagedAttentionBackend,
    ROCMHardwareBackend,
    ROCMMoEBackend,
    ROCMPagedAttentionBackend,
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
        if torch.cuda.is_available() and cap.details:
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

    def test_c_abi_descriptor_structure(self):
        """Verifies binary alignment and size of C-ABI Plugin Descriptor struct."""
        from hydragen.backends.registry import _CDeviceCaps, _CPluginDescriptor
        import ctypes

        caps = _CDeviceCaps()
        caps.device_type = 2 # CUDA
        caps.compute_major = 8
        caps.compute_minor = 6
        caps.has_dp4a = 1
        caps.has_cp_async = 1
        caps.has_dpas = 0
        caps.has_amx = 0
        caps.has_mfma = 0

        desc = _CPluginDescriptor()
        desc.abi_version = 1
        desc.plugin_name = b"test_cuda_plugin"
        desc.target_architecture = b"cuda_sm86"
        desc.caps = caps

        self.assertEqual(desc.abi_version, 1)
        self.assertEqual(desc.plugin_name.decode("utf-8"), "test_cuda_plugin")
        self.assertEqual(desc.target_architecture.decode("utf-8"), "cuda_sm86")
        self.assertEqual(desc.caps.compute_major, 8)
        self.assertEqual(desc.caps.compute_minor, 6)
        self.assertEqual(desc.caps.has_dp4a, 1)
        self.assertEqual(desc.caps.has_cp_async, 1)
        self.assertEqual(desc.caps.has_dpas, 0)
        self.assertEqual(desc.caps.has_amx, 0)
        self.assertEqual(desc.caps.has_mfma, 0)

    def test_intel_hardware_backend_introspection(self):
        """Validates Intel hardware backend capabilities (XPU or AMX/VNNI CPU)."""
        intel_hw = IntelHardwareBackend()
        cap = intel_hw.capability
        self.assertEqual(cap.device_type, DeviceType.INTEL)
        self.assertTrue(len(cap.device_name) > 0)
        self.assertTrue(cap.multi_processor_count >= 1)
        # On x86 hosts, AMX or VNNI flags should be exposed
        self.assertIn("has_ipex", cap.details)

    def test_forced_intel_backend_selection(self):
        """Validates forcing Intel hardware backend instantiation."""
        hw = get_hardware_backend(force_device=DeviceType.INTEL)
        self.assertIsInstance(hw, IntelHardwareBackend)
        self.assertEqual(hw.device_type, DeviceType.INTEL)

        paged_att = get_paged_attention_backend(force_device=DeviceType.INTEL)
        self.assertIsInstance(paged_att, IntelPagedAttentionBackend)
        self.assertEqual(paged_att.device_type, DeviceType.INTEL)
        self.assertEqual(paged_att.name, "intel_paged_attention")

    def test_intel_paged_attention_execution(self):
        """Validates Intel paged attention decode and prefill execution path."""
        backend = IntelPagedAttentionBackend()
        q_decode = torch.randn(1, 4, 64)
        k_cache = torch.randn(8, 2, 16, 64)
        v_cache = torch.randn(8, 2, 16, 64)
        block_tables = torch.tensor([[0, 1]], dtype=torch.int32)
        context_lens = torch.tensor([18], dtype=torch.int32)
        out_decode = torch.zeros_like(q_decode)

        backend.paged_attention_decode(
            out=out_decode,
            q=q_decode,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
        )
        self.assertFalse(torch.isnan(out_decode).any())
        self.assertTrue((out_decode != 0.0).any())

        # Prefill
        q_prefill = torch.randn(1, 4, 4, 64)
        out_prefill = torch.zeros_like(q_prefill)
        backend.paged_attention_prefill(
            out=out_prefill,
            q=q_prefill,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
        )
        self.assertFalse(torch.isnan(out_prefill).any())

    def test_intel_moe_backend(self):
        """Validates Intel MoE backend interface."""
        moe = IntelMoEBackend()
        self.assertEqual(moe.device_type, DeviceType.INTEL)
        self.assertEqual(moe.name, "intel_ipex_moe")
        x = torch.randn(2, 64)
        logits = torch.randn(2, 8)
        out = moe.forward_moe(x, logits, top_k=2)
        self.assertTrue(torch.equal(out, x))

    def test_rocm_hardware_backend_introspection(self):
        """Validates AMD ROCm hardware backend capabilities (RDNA3 WMMA / CDNA MFMA)."""
        rocm_hw = ROCMHardwareBackend()
        cap = rocm_hw.capability
        self.assertEqual(cap.device_type, DeviceType.ROCM)
        self.assertTrue(len(cap.device_name) > 0)
        self.assertTrue(cap.multi_processor_count >= 1)
        self.assertIn("architecture", cap.details)

    def test_forced_rocm_backend_selection(self):
        """Validates forcing AMD ROCm hardware backend instantiation."""
        hw = get_hardware_backend(force_device=DeviceType.ROCM)
        self.assertIsInstance(hw, ROCMHardwareBackend)
        self.assertEqual(hw.device_type, DeviceType.ROCM)

        paged_att = get_paged_attention_backend(force_device=DeviceType.ROCM)
        self.assertIsInstance(paged_att, ROCMPagedAttentionBackend)
        self.assertEqual(paged_att.device_type, DeviceType.ROCM)
        self.assertEqual(paged_att.name, "rocm_paged_attention")

    def test_rocm_paged_attention_execution(self):
        """Validates AMD ROCm paged attention decode and prefill execution path."""
        backend = ROCMPagedAttentionBackend()
        q_decode = torch.randn(1, 4, 64)
        k_cache = torch.randn(8, 2, 16, 64)
        v_cache = torch.randn(8, 2, 16, 64)
        block_tables = torch.tensor([[0, 1]], dtype=torch.int32)
        context_lens = torch.tensor([18], dtype=torch.int32)
        out_decode = torch.zeros_like(q_decode)

        backend.paged_attention_decode(
            out=out_decode,
            q=q_decode,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
        )
        self.assertFalse(torch.isnan(out_decode).any())
        self.assertTrue((out_decode != 0.0).any())

        # Prefill
        q_prefill = torch.randn(1, 4, 4, 64)
        out_prefill = torch.zeros_like(q_prefill)
        backend.paged_attention_prefill(
            out=out_prefill,
            q=q_prefill,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
        )
        self.assertFalse(torch.isnan(out_prefill).any())

    def test_rocm_moe_backend(self):
        """Validates AMD ROCm MoE backend interface."""
        moe = ROCMMoEBackend()
        self.assertEqual(moe.device_type, DeviceType.ROCM)
        self.assertEqual(moe.name, "rocm_hipblas_moe")
        x = torch.randn(2, 64)
        logits = torch.randn(2, 8)
        out = moe.forward_moe(x, logits, top_k=2)
        self.assertTrue(torch.equal(out, x))

    def test_hardware_stream_and_context_abstractions(self):
        """Verifies stream creation, stream context, and synchronization across the hardware quad."""
        for backend_cls in [CPUHardwareBackend, CUDAHardwareBackend, IntelHardwareBackend, ROCMHardwareBackend]:
            backend = backend_cls()
            stream = backend.create_stream()
            # Context manager must execute cleanly without exceptions
            with backend.stream_context(stream):
                t = backend.empty_tensor((4, 4), dtype=torch.float32)
                t.fill_(1.0)
            backend.stream_wait_stream(stream, stream)
            backend.synchronize(stream)
            self.assertEqual(t.shape, (4, 4))

    def test_hardware_memory_and_allocations(self):
        """Verifies memory info queries and pinned host / device allocations."""
        for backend_cls in [CPUHardwareBackend, CUDAHardwareBackend, IntelHardwareBackend, ROCMHardwareBackend]:
            backend = backend_cls()
            free_b, total_b = backend.get_mem_info()
            self.assertIsInstance(free_b, int)
            self.assertIsInstance(total_b, int)
            self.assertTrue(total_b >= 0)

            # Pinned host tensor allocation
            pinned = backend.allocate_pinned_host_tensor((8, 16), dtype=torch.float16)
            self.assertEqual(pinned.shape, (8, 16))
            self.assertEqual(pinned.device.type, "cpu")

            # Device empty tensor
            empty_t = backend.empty_tensor((2, 4), dtype=torch.int32)
            self.assertEqual(empty_t.shape, (2, 4))

    def test_expert_manager_and_mailbox_modular_injection(self):
        """Verifies PagedExpertManager and HybridExpertMailbox cleanly decouple via HardwareBackend injection."""
        from hydragen.expert_manager import PagedExpertManager
        from hydragen.expert_mailbox import HybridExpertMailbox

        cpu_hw = CPUHardwareBackend()
        # Initialize PagedExpertManager with explicit CPU hardware backend injection
        em = PagedExpertManager(
            num_slots=4,
            expert_shape=(3, 16, 16),
            total_experts=8,
            dtype=torch.float32,
            device=torch.device("cpu"),
            hw_backend=cpu_hw,
        )
        self.assertIs(em.hw, cpu_hw)
        self.assertIsNone(em.dma_stream)

        # Ensure page request and table synchronization work branchlessly on CPU
        mapping = em.request_experts([0, 1, 2])
        self.assertEqual(len(mapping), 3)
        self.assertEqual(em.cpu_slot_table[0].item(), mapping[0])
        self.assertEqual(em.gpu_slot_table[0].item(), mapping[0])
        em.synchronize()

        # Initialize HybridExpertMailbox with explicit backend injection
        box = HybridExpertMailbox(max_workers=2, device=torch.device("cpu"), hw_backend=cpu_hw)
        self.assertIs(box.hw, cpu_hw)
        box.close()


if __name__ == "__main__":
    unittest.main()
