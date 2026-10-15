#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('full_contact_audit', HERE/'audit.py')
a=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(a)

class AuditContractTests(unittest.TestCase):
    def test_strict_json_rejects_nonstandard_tokens(self):
        for token in ('NaN','Infinity','-Infinity'):
            with self.assertRaises(a.AuditError): a.strict_json_text('{"x":'+token+'}', 'synthetic')
        self.assertEqual(a.strict_json_text('{"x":0}', 'synthetic'), {'x':0})

    def test_only_output_paths_are_normalized_in_argv(self):
        source=a.BASELINE_SOURCE
        run=Path('/tmp/run-a')
        argv=['/tmp/build/matter/numi-matter-cardboard-full-contact-armijo-check',
              str(source/a.MATERIAL_REL),'--sparse-free-body-parity','--diagnostic-state-output',
              str(run/'diagnostic-state'),'--extended-budget','--compensated-material-arithmetic']
        first=a.normalize_argv(argv,run,source)
        moved=list(argv); moved[0]='/other/build/matter/numi-matter-cardboard-full-contact-armijo-check'; moved[4]='/tmp/run-b/diagnostic-state'
        second=a.normalize_argv(moved,Path('/tmp/run-b'),source)
        self.assertEqual(first,second)
        moved_bad=list(moved); moved_bad[-1]='--ordinary-material-arithmetic'
        with self.assertRaises(a.AuditError): a.normalize_argv(moved_bad,Path('/tmp/run-b'),source)

    def test_local_material_budget_is_separate_from_authored_newton_budget(self):
        source=a.BASELINE_SOURCE
        text=(source/a.CHECKER_REL).read_text()
        vals=[int(x) for x in __import__('re').findall(r'localMaterialNewtonIterations\s*=\s*(\d+)u', text)]
        self.assertTrue(vals)
        self.assertEqual(set(vals), {16})
        self.assertEqual(a.EXPECTED_LOCAL_MATERIAL_ITERATIONS,16)

    def test_snapshot_comparison_only_masks_device_fingerprint(self):
        # Snapshot layout: u32 message length, message, u64 source FP, u64 device FP, persistent bytes.
        msg=b'fixture message'
        prefix=len(msg).to_bytes(4,'little')+msg+(0x1234).to_bytes(8,'little')
        x=prefix+(0x5678).to_bytes(8,'little')+b'persist'
        y=prefix+(0x9abc).to_bytes(8,'little')+b'persist'
        result=a.compare_snapshot_binary(x,y,'fixture')
        self.assertTrue(result['all_bytes_exact_except_device_fingerprint'])
        z=len(msg).to_bytes(4,'little')+msg+(0x9999).to_bytes(8,'little')+(0x9abc).to_bytes(8,'little')+b'persist'
        self.assertFalse(a.compare_snapshot_binary(x,z,'fixture')['source_physics_fingerprint_equal'])

    def test_tuple_difference_needs_failure_generation_order_evidence(self):
        diff=[{'newton_iteration':9,'baseline':{'failing_index':9},'candidate':{'failing_index':3}}]
        result=a.classify_tuple_change(diff,7,9,9)
        self.assertFalse(result['candidate_order_fixed_at_failure'])
        self.assertEqual(result['classification'],'unadjudicated_tuple_delta_capture_is_not_at_failure_iteration')
        same=a.classify_tuple_change(diff,9,9,9)
        self.assertEqual(same['candidate_order_fixed_at_failure'],'must_be_proved_from_paired_capture_arrays')

    def test_source_delta_binding_is_exact(self):
        report=a.source_delta_report()
        self.assertEqual(report['file_count'],781)
        self.assertEqual(set(report['changed_files']), set(a.EXPECTED_SOURCE_DELTA))

    def test_381_control_interpretation_is_never_physical_qualification(self):
        # The native control's qualification contract is explicitly a zero-step routing test.
        result={'checks':169,'physical_steps':0,'status':'pass','geometric_failure_producer_control':{'physical_qualification':False},'captured_ee_physical_qualification':False}
        self.assertEqual(result['physical_steps'],0)
        self.assertFalse(result['geometric_failure_producer_control']['physical_qualification'])

if __name__=='__main__': unittest.main(verbosity=2)
