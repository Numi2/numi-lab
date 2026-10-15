import re
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
FIXED_RUNTIME = PACKAGE / "fixed/matter/src/runtime.mm"


def pack_environment_major(source, environments, modes, nodes, padded_modes):
    """Mirror runtime's segmented MTL blits into compact [E, modes, nodes]."""
    source_stride = padded_modes * nodes
    segment_count = environments
    segment_bytes = modes * nodes
    output = []
    for environment in range(segment_count):
        source_offset = environment * source_stride
        output.extend(source[source_offset:source_offset + segment_bytes])
    return output


class CaptureSegmentLayoutTests(unittest.TestCase):
    def test_augmented_environment_stride_matches_snapshot_compaction(self):
        environments, base_modes, nodes = 2, 2, 3
        padded_modes = base_modes + 1
        source = [100 * e + 10 * m + n
                  for e in range(environments)
                  for m in range(padded_modes)
                  for n in range(nodes)]
        rich = pack_environment_major(source, environments, base_modes,
                                      nodes, padded_modes)
        snapshot = [source[(e * padded_modes + m) * nodes + n]
                    for e in range(environments)
                    for m in range(base_modes)
                    for n in range(nodes)]
        self.assertEqual(rich, snapshot)
        # The old one-shot contiguous blit incorrectly includes env0's pad row.
        self.assertNotEqual(rich, source[:len(rich)])

    def test_unaugmented_stride_preserves_compact_layout(self):
        environments, modes, nodes = 3, 4, 5
        source = list(range(environments * modes * nodes))
        rich = pack_environment_major(source, environments, modes, nodes, modes)
        self.assertEqual(rich, source)

    def test_runtime_uses_segmented_blits_only_for_padded_environment_arrays(self):
        text = re.sub(r"\s+", " ", FIXED_RUNTIME.read_text())
        required = (
            "sourceSegmentBytes",
            "sourceSegmentStride",
            "sourceSegmentCount",
            "effectiveModeStride * nodeBytes",
            "0u, effectiveSegmentBytes, effectiveSourceStride, envCount",
            "segment * array.sourceSegmentStride",
            "segment * array.sourceSegmentBytes",
            "sourceSegmentCount * sourceSegmentBytes == byteCount",
        )
        missing = [token for token in required if token not in text]
        if missing:
            raise AssertionError(f"runtime segmented-capture contract missing: {missing}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
