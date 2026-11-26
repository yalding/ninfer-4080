import unittest

import torch

from tools.artifact.layouts import encode_row_split, row_split_geometry
from tools.convert.common.quantize import quantize_and_encode
from tools.convert.inkling2_mini import inventory
from tools.convert.inkling2_mini.verify import (
    VerificationError,
    _dequantize_rows,
    _verify_q3_canonical,
)


class VerifyTest(unittest.TestCase):
    def test_q6_sample_rows_preserve_high_plane(self):
        shape = (5, 256)
        codes = ((torch.arange(5 * 256).reshape(5, 4, 64) * 7 + 13) % 64 - 32).to(torch.int8)
        scales = torch.arange(1, 21, dtype=torch.float16).reshape(5, 4) / 8
        payload = encode_row_split(codes, scales, "Q6G64_F16S", shape)
        rows = [4, 0, 2]
        actual = _dequantize_rows(memoryview(payload), "Q6G64_F16S", shape, rows, torch.device("cpu"))
        expected = (codes.float() * scales.float().unsqueeze(-1)).reshape(shape)[rows]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_q3_canonical_and_corruption(self):
        generator = torch.Generator().manual_seed(411)
        source = torch.randn((3, 2304), generator=generator)
        source[0, :128] = 0
        source[1, :128] *= 1e-8  # FP16 scale underflow uses the minimum subnormal.
        payload = quantize_and_encode(source, "Q3G128_F16S", device="cpu")
        actual = _dequantize_rows(memoryview(payload), "Q3G128_F16S", tuple(source.shape), [0, 1, 2], torch.device("cpu"))
        _verify_q3_canonical(actual, source, "correct payload")
        corrupt = bytearray(payload)
        geometry = row_split_geometry("Q3G128_F16S", source.shape)
        corrupt[geometry.base_row_bytes + 80] ^= 1
        changed = _dequantize_rows(memoryview(corrupt), "Q3G128_F16S", tuple(source.shape), [0, 1, 2], torch.device("cpu"))
        with self.assertRaisesRegex(VerificationError, "canonical"):
            _verify_q3_canonical(changed, source, "corrupted payload")


    def test_dflash2_inventory_contract(self):
        self.assertEqual(len(inventory.DFLASH2_TENSOR_SPECS), 76)
        self.assertEqual(
            len(inventory.object_specs(include_dflash2=True)),
            len(inventory.OBJECT_SPECS) + 76,
        )
        counts = inventory.format_counts(inventory.DFLASH2_TENSOR_SPECS)
        self.assertEqual(counts[inventory.BF16], 45)
        self.assertEqual(counts[inventory.Q4], 31)
        by_name = {s.name: s for s in inventory.DFLASH2_TENSOR_SPECS}
        self.assertEqual(by_name["dflash2/fc"].shape, (2304, 9216))
        self.assertEqual(by_name["dflash2/layers/0/attention_conv/base_kernel"].shape, (2, 2, 2304))
        self.assertEqual(by_name["dflash2/layers/4/mlp/gate_up"].shape, (13824, 2304))
        self.assertEqual(by_name["dflash2/selector/predecessor_codebook"].shape, (201024, 256))


if __name__ == "__main__":
    unittest.main()
