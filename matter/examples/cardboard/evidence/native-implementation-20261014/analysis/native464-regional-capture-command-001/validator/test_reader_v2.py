#!/usr/bin/env python3
import unittest
import read_regional_capture_v2 as r
import test_reader_v1 as v1_fixtures

class StrictRegionalV2Tests(unittest.TestCase):
    def test_augmented_tag_matching_first_three_still_unadmitted(self):
        row=r._augmented_cycle_admission((1,2,48,8),(1,2,48),1,1)
        self.assertTrue(row['root_substep_newton_match'])
        self.assertEqual(row['producer_restart_cycle'],8)
        self.assertIsNone(row['consumer_restart_cycle'])
        self.assertFalse(row['augmented_data_admissible_for_same_newton'])
        self.assertEqual(row['reason'],'consumed_restart_cycle_not_captured_independently')

    def test_full_reader_overrides_v1_cycle_prefix_admission(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            path=v1_fixtures.make_capture(Path(temp), augmented=True)
            result=r.read_capture(path)
            self.assertTrue(result['identity_joined'])
            self.assertFalse(result['augmented_generation'][0]['augmented_data_admissible_for_same_newton'])
            self.assertEqual(result['augmented_generation'][0]['reason'],
                             'consumed_restart_cycle_not_captured_independently')

    def test_producer_cycle_mismatch_does_not_become_current(self):
        row=r._augmented_cycle_admission((1,2,47,8),(1,2,48),1,1)
        self.assertFalse(row['root_substep_newton_match'])
        self.assertFalse(row['augmented_data_admissible_for_same_newton'])

    def test_validity_words_do_not_supply_missing_consumer_cycle(self):
        for valid in (0,1):
            row=r._augmented_cycle_admission((4,3,9,11),(4,3,9),valid,valid)
            self.assertFalse(row['augmented_data_admissible_for_same_newton'])

    def test_pivot_permutation_exact(self):
        self.assertTrue(r._is_permutation((2,0,1),3))
        self.assertFalse(r._is_permutation((2,2,1),3))
        self.assertFalse(r._is_permutation((0,1),3))

    def test_graph_checksum_matches_source_word_order(self):
        arrays={name:memoryview(bytes.fromhex(value)) for name,value in {
            'graph_patches':'01000000020000000300000004000000',
            'graph_rows':'05000000060000000700000008000000',
            'graph_patch_nodes':'09000000',
            'graph_node_patch_offsets':'0a000000',
            'graph_node_patch_values':'0b000000',
        }.items()}
        expected=2166136261
        for name in ('graph_patches','graph_rows','graph_patch_nodes','graph_node_patch_offsets','graph_node_patch_values'):
            for off in range(0,len(arrays[name]),4):
                w=int.from_bytes(arrays[name][off:off+4],'little')
                expected=((expected^w)*16777619)&0xffffffff
        self.assertEqual(r._fnv32_arrays(arrays,tuple(arrays)),expected)

    def test_checksum_rejects_misaligned_words(self):
        with self.assertRaises(r.CaptureError):
            r._fnv32_arrays({'x':memoryview(b'123')},('x',))

if __name__=='__main__': unittest.main()
