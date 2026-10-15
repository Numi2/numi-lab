#!/usr/bin/env python3
"""CPU-only fail-closed tests, including synthetic native-result parser cases."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('pair_package_verifier', HERE/'verify_pair_package.py')
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)

def fbits(value):
    return f'{struct.unpack("<I",struct.pack("<f",float(value)))[0]:08x}'

def zero_result_for_case(case):
    # Synthetic parser fixture only. The records deliberately carry a
    # stationary-certified-miss summary; no physical geometry claim is made.
    thickness=float(case['thickness_m'])
    eff=max(thickness,1.0e-8)
    prod=[
        [fbits(0),fbits(1),fbits(0),fbits(1)],
        [fbits(0),fbits(0),fbits(0),fbits(0)],
        [fbits(0),fbits(0),fbits(0),fbits(0)],
        [fbits(0),fbits(0),fbits(0),fbits(0)],
    ]
    clone=[list(x) for x in prod]
    # Host compares only record-3 x. These reserved/diagnostic words differ
    # intentionally to exercise that exact comparator contract.
    clone[3][1:]=[fbits(1),fbits(4),fbits(eff)]
    clone.append([fbits(0),fbits(123),fbits(456),fbits(789)])
    common={
        'hit':False,'certified':True,'toi':0.0,'separation_m':1.0,
        'normal':[0.0,0.0,0.0],'feature_parameter0':0.0,
        'witness_point':[0.0,0.0,0.0],'feature_parameter1':0.0,'feature_parameter2':0.0,
    }
    production={**common,'raw_float4_bits':prod}
    trace={**common,'iteration_count':1,'terminal_code':4,
        'effective_thickness_m':eff,'speed_bound_m':0.0,'raw_float4_bits':clone}
    return {
        **{k:case[k] for k in ('name','kind_id','kind','feature_a','feature_b','capture','primitive_pair','candidate_slot','candidate_identity')},
        'production_result':production,
        'trace_clone_result':trace,
        'production_and_trace_clone_bitwise_equal':True,
        'native_conservative_advance_trace':[{'iteration':0,'time':0.0,'distance_m':1.0,'step':0.0,'terminal_code':4}],
    }

def synthetic_native_result(prepared_path):
    p=VERIFY.strict_json(prepared_path)
    return {
        'schema':'numi.cardboard.source103_pair_ccd_native_results.v1',
        'source_snapshot':p['source_snapshot'],
        'source_contact_sha256':p['source_contact_sha256'],
        'source_common_sha256':p['source_common_sha256'],
        'source_main_shader_sha256':p['source_main_shader_sha256'],
        'prepared_manifest_sha256':VERIFY.sha(prepared_path),
        'cases_bin_sha256':p['case_contract']['input_sha256'],
        'metallib_path':str((HERE/'compile/diagnostic_pair.metallib').resolve()),
        'metallib_sha256':VERIFY.sha(HERE/'compile/diagnostic_pair.metallib'),
        'physical_transactions_executed':0,
        'diagnostic_execution_only':True,
        'production_vs_trace_clone_all_bitwise_equal':True,
        'cases':[zero_result_for_case(c) for c in p['cases']],
    }

class PackageContractTests(unittest.TestCase):
    def setUp(self):
        self.prepared_path=HERE/'prepared/prepared.json'
        self.prepared,self.summary=VERIFY.validate_prepared(self.prepared_path)

    def _validate_synthetic(self, mutate=None):
        data=synthetic_native_result(self.prepared_path)
        if mutate:
            mutate(data)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'native.json'
            path.write_text(json.dumps(data,allow_nan=False))
            return VERIFY.validate_native(path,self.prepared_path,self.prepared,copy.deepcopy(self.summary))

    def test_current_prepared_package_and_speed_bounds(self):
        self.assertEqual(self.summary['case_count'],30)
        self.assertEqual(self.summary['case_counts'],{'vertex_triangle':12,'edge_edge':18})
        self.assertGreater(self.summary['speed_bound_ratio_ranges']['vertex_triangle']['max'],6.0)
        self.assertGreater(self.summary['speed_bound_ratio_ranges']['edge_edge']['max'],2.0)
        self.assertFalse(self.summary['native_result_admitted'])

    def test_source_binding_tamper_rejected(self):
        original=VERIFY.strict_json(self.prepared_path)
        original['source_contact_sha256']='0'*64
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'prepared.json'
            path.write_text(json.dumps(original))
            (Path(temp)/'cases.bin').write_bytes((HERE/'prepared/cases.bin').read_bytes())
            with self.assertRaisesRegex(ValueError,'frozen source hash mismatch'):
                VERIFY.validate_prepared(path)

    def test_binary_tamper_rejected(self):
        original=VERIFY.strict_json(self.prepared_path)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            raw=bytearray((HERE/'prepared/cases.bin').read_bytes());raw[0]^=1
            (root/'cases.bin').write_bytes(raw)
            (root/'prepared.json').write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError,'input binary hash mismatch'):
                VERIFY.validate_prepared(root/'prepared.json')

    def test_speed_bound_claim_tamper_rejected(self):
        original=VERIFY.strict_json(self.prepared_path)
        original['cases'][0]['cpu_fp64_reference']['speed_bound_comparison']['galilean_relative_lipschitz_bound_m']*=0.5
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'cases.bin').write_bytes((HERE/'prepared/cases.bin').read_bytes())
            (root/'prepared.json').write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError,'relative bound mismatch'):
                VERIFY.validate_prepared(root/'prepared.json')

    def test_synthetic_native_result_positive_and_exact_host_comparator(self):
        result=self._validate_synthetic()
        self.assertTrue(result['production_vs_trace_clone_all_bitwise_equal'])

    def test_synthetic_native_result_rejects_wrong_host_flag(self):
        def mutate(d): d['cases'][0]['production_and_trace_clone_bitwise_equal']=False
        with self.assertRaisesRegex(ValueError,'host equality flag disagrees'):
            self._validate_synthetic(mutate)

    def test_synthetic_native_result_rejects_compared_bit_mismatch_even_if_flag_true(self):
        def mutate(d): d['cases'][0]['trace_clone_result']['raw_float4_bits'][0][2]=fbits(0.25)
        with self.assertRaisesRegex(ValueError,'host equality flag disagrees'):
            self._validate_synthetic(mutate)

    def test_synthetic_native_result_rejects_malformed_word_width(self):
        def mutate(d): d['cases'][0]['production_result']['raw_float4_bits'][0][0]='0000000'
        with self.assertRaisesRegex(ValueError,'malformed 32-bit float'):
            self._validate_synthetic(mutate)

    def test_synthetic_native_result_rejects_wrong_metallib_hash(self):
        def mutate(d): d['metallib_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'metallib hash mismatch'):
            self._validate_synthetic(mutate)

    def test_synthetic_native_result_rejects_wrong_trace_count(self):
        def mutate(d): d['cases'][0]['native_conservative_advance_trace']=[]
        with self.assertRaisesRegex(ValueError,'trace count differs'):
            self._validate_synthetic(mutate)

if __name__=='__main__':
    unittest.main(verbosity=2)
