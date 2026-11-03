# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Pluggable Speculative Drafter Architecture (MTP vs DFlash2)."""

import os
import unittest
import torch

from hydragen.backends.base import DeviceCapability, DeviceType, HardwareBackend
from hydragen.drafters import (
    DisabledDrafter,
    DFlash2Drafter,
    DraftCandidate,
    DraftingMode,
    MtpDrafter,
    get_speculative_drafter,
    resolve_drafting_mode,
)


class TestSpeculativeDrafters(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        if "HYDRAGEN_DRAFTER" in os.environ:
            del os.environ["HYDRAGEN_DRAFTER"]

    def test_mtp_drafter_lifecycle(self):
        """Validates MTP drafter prefill pre-conditioning, drafting, and feedback."""
        drafter = MtpDrafter(n_embd=256, hc=2, n_head_kv=2, head_dim=64, max_context=128, device=torch.device("cpu"))
        self.assertEqual(drafter.mode, DraftingMode.MTP)
        self.assertEqual(drafter.name, "mtp_autoregressive_drafter")
        self.assertTrue(drafter.is_available())

        # Test prefill step
        prompt_tokens = torch.tensor([101, 205, 309, 412], dtype=torch.long)
        hidden_states = torch.randn(4, 2 * 256)
        drafter.prefill_step(prompt_tokens, hidden_states)

        # Test speculative drafting
        candidate = drafter.draft(anchor_token=412, hidden_state=hidden_states[-1:], n_tokens=4)
        self.assertIsInstance(candidate, DraftCandidate)
        self.assertEqual(len(candidate.tokens), 4)
        self.assertEqual(candidate.source, "mtp")

        # Test feedback
        drafter.accept_feedback(3)
        self.assertAlmostEqual(drafter.acceptance_rate, 0.75, places=2)

    def test_dflash2_block_drafter(self):
        """Validates DFlash2 non-causal block drafting and metadata."""
        drafter = DFlash2Drafter(n_embd=256, hc=2, block_tokens=7, device=torch.device("cpu"))
        self.assertEqual(drafter.mode, DraftingMode.DFLASH2)
        self.assertEqual(drafter.name, "dflash2_block_drafter")
        self.assertTrue(drafter.is_available())

        hidden_state = torch.randn(1, 2 * 256)
        candidate = drafter.draft(anchor_token=1000, hidden_state=hidden_state, n_tokens=7)
        self.assertIsInstance(candidate, DraftCandidate)
        self.assertEqual(len(candidate.tokens), 7)
        self.assertEqual(candidate.source, "dflash2")
        self.assertTrue(candidate.metadata.get("single_pass"))
        self.assertEqual(candidate.metadata.get("vram_freed_mib"), 367)

        drafter.accept_feedback(5)
        self.assertAlmostEqual(drafter.acceptance_rate, 5 / 7, places=2)

    def test_disabled_drafter(self):
        """Validates null drafter for mode=OFF."""
        drafter = DisabledDrafter()
        self.assertEqual(drafter.mode, DraftingMode.OFF)
        self.assertFalse(drafter.is_available())

        candidate = drafter.draft(anchor_token=100, hidden_state=torch.randn(1, 256))
        self.assertEqual(len(candidate.tokens), 0)
        self.assertEqual(candidate.confidence, 0.0)

    def test_resolve_drafting_mode_auto(self):
        """Verifies architecture-aware auto resolution."""
        # Mock Ampere Hardware
        class MockAmpereBackend(HardwareBackend):
            @property
            def device_type(self) -> DeviceType:
                return DeviceType.CUDA
            @property
            def capability(self) -> DeviceCapability:
                return DeviceCapability(
                    device_type=DeviceType.CUDA,
                    device_name="NVIDIA GeForce RTX 3080 Ti (sm_86)",
                    details={"architecture": "Ampere"},
                )
            def get_paged_attention_backend(self): return None
            def get_moe_backend(self): return None
            def get_ipc_backend(self): return None

        mode = resolve_drafting_mode(DraftingMode.AUTO, hardware_backend=MockAmpereBackend())
        self.assertEqual(mode, DraftingMode.MTP)

        # Mock Blackwell Hardware
        class MockBlackwellBackend(HardwareBackend):
            @property
            def device_type(self) -> DeviceType:
                return DeviceType.CUDA
            @property
            def capability(self) -> DeviceCapability:
                return DeviceCapability(
                    device_type=DeviceType.CUDA,
                    device_name="NVIDIA GeForce RTX 5070 (sm_120)",
                    details={"architecture": "Blackwell"},
                )
            def get_paged_attention_backend(self): return None
            def get_moe_backend(self): return None
            def get_ipc_backend(self): return None

        mode_blackwell = resolve_drafting_mode(DraftingMode.AUTO, hardware_backend=MockBlackwellBackend())
        self.assertEqual(mode_blackwell, DraftingMode.DFLASH2)

    def test_env_override_drafting_mode(self):
        """Verifies HYDRAGEN_DRAFTER environment variable override."""
        os.environ["HYDRAGEN_DRAFTER"] = "dflash2"
        self.assertEqual(resolve_drafting_mode(), DraftingMode.DFLASH2)

        os.environ["HYDRAGEN_DRAFTER"] = "mtp"
        self.assertEqual(resolve_drafting_mode(), DraftingMode.MTP)

        os.environ["HYDRAGEN_DRAFTER"] = "off"
        self.assertEqual(resolve_drafting_mode(), DraftingMode.OFF)

    def test_factory_instantiation(self):
        """Verifies get_speculative_drafter creates correct implementations."""
        drafter_mtp = get_speculative_drafter(DraftingMode.MTP, n_embd=128, hc=1, n_head_kv=1, head_dim=32, max_context=64, device=torch.device("cpu"))
        self.assertIsInstance(drafter_mtp, MtpDrafter)

        drafter_dflash = get_speculative_drafter(DraftingMode.DFLASH2, n_embd=128, hc=1, device=torch.device("cpu"))
        self.assertIsInstance(drafter_dflash, DFlash2Drafter)

        drafter_off = get_speculative_drafter(DraftingMode.OFF)
        self.assertIsInstance(drafter_off, DisabledDrafter)


if __name__ == "__main__":
    unittest.main()
