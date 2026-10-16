import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / 'prepare_resume.py'
spec_obj = importlib.util.spec_from_file_location('prepare_resume', MODULE)
mod = importlib.util.module_from_spec(spec_obj)
spec_obj.loader.exec_module(mod)

class ResumeContractTests(unittest.TestCase):
    def test_exact_resume_flags_and_physics_argv(self):
        p = Path('/Users/home/cardboard-implementation-20261014/analysis/native471-checkpoint-resume-command-001/native-471-command.json')
        d = json.loads(p.read_text())
        native = d['native_argv']
        self.assertEqual(native.count('--coupon-resume-checkpoint'), 1)
        self.assertEqual(native.count('--coupon-resume-metadata'), 1)
        self.assertNotIn('--coupon-checkpoint-after-control', native)
        self.assertEqual(native[native.index('--coupon-resume-checkpoint')+1], d['checkpoint_contract']['checkpoint'])
        self.assertEqual(native[native.index('--coupon-resume-metadata')+1], d['checkpoint_contract']['resume_metadata'])
        self.assertEqual(d['environment_overrides']['NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR'], '20:0:63:1:0:0')

    def test_properties_reject_duplicate_and_malformed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'props'; p.write_text('a=1\na=2\n')
            with self.assertRaises(ValueError): mod.parse_properties(p)
            p.write_text('not-a-property\n')
            with self.assertRaises(ValueError): mod.parse_properties(p)

    def test_prefix_validator_accepts_only_exact_accepted_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); case=root/'source470'; out=case/'output'; ckdir=out/'checkpoints'
            ckdir.mkdir(parents=True)
            cp=ckdir/'coupon-control-000019.nmcb'; cp.write_bytes(b'accepted snapshot bytes')
            observations=out/'observations.csv'; observations.write_bytes(b'control,accepted\n19,1\n')
            protocol=out/'manifest.json'; protocol.write_bytes(b'{"protocol":"frozen"}\n')
            prefix=ckdir/'coupon-control-000019.prefix'
            prefix.write_text('schema=numi.cardboard.accepted-coupon-prefix.v1\n'
                f'source_run_directory={out.resolve()}\naccepted_control_index=19\n'
                f'observations_prefix_bytes={observations.stat().st_size}\n'
                f'observations_prefix_sha256={mod.sha(observations)}\n'
                f'file=observations.csv|{observations.stat().st_size}|{mod.sha(observations)}|1\n')
            meta=ckdir/'coupon-control-000019.resume'
            fields={
                'schema':'numi.cardboard.accepted-coupon-resume.v2',
                'accepted_control_index':'19','next_control_index':'20',
                'source_prefix_control_count':'20','environment_count':'2',
                'both_environments_accepted':'true','checkpoint_sha256':mod.sha(cp),
                'checkpoint_bytes':str(cp.stat().st_size),'prefix_manifest_sha256':mod.sha(prefix),
                'prefix_manifest_file':prefix.name,'source_run_directory':str(out.resolve()),
                'protocol_manifest_sha256':mod.sha(protocol),
                'source_runner_case_directory':str(case.resolve()),
            }
            meta.write_text(''.join(f'{k}={v}\n' for k,v in fields.items()))
            (out/'result.json').write_text(json.dumps({'schema':'numi.cardboard.probe-result.v1','status':'accepted_checkpoint_prefix_saved','accepted_steps':20,'physical_validation':False}))
            (out/'coupon-checkpoint-prefix.json').write_text(json.dumps({
                'schema':'numi.cardboard.coupon-checkpoint-prefix.v1','status':'accepted_checkpoint_prefix_saved',
                'accepted_control_index':19,'next_control_index':20,
                'checkpoint_file':'checkpoints/coupon-control-000019.nmcb','checkpoint_sha256':mod.sha(cp),
                'metadata_file':'checkpoints/coupon-control-000019.resume','metadata_sha256':mod.sha(meta),
                'full_protocol_complete':False,'trajectory_parity_established':False,'physical_validation':False}))
            (case/'receipt.json').write_text(json.dumps({'schema':'numi.cardboard.native-run-receipt.v2','execution_started':True,'exit_code':0,'timed_out':False,'source_inputs_unchanged':True,'changed_inputs':[]}))
            (case/'diagnostic-captures.json').write_text(json.dumps({'status':'validated_with_pending_checkpoint_targets','pending_captures':[{'key':'NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH','target_control':20,'accepted_control':19,'status':'not_yet_visited_at_accepted_checkpoint','positive_qualification_allowed':False}]}))
            s={'source_output':str(out),'source_case':str(case),'checkpoint_contract':{'checkpoint':str(cp),'resume_metadata':str(meta),'prefix_manifest':str(prefix)}}
            got=mod.verify_prefix(s)
            self.assertEqual(got['source_receipt_exit_code'],0)
            self.assertTrue(got['capture_pending'])
            (case/'receipt.json').write_text(json.dumps({'schema':'numi.cardboard.native-run-receipt.v2','execution_started':True,'exit_code':0,'timed_out':True,'source_inputs_unchanged':True,'changed_inputs':[]}))
            with self.assertRaises(ValueError): mod.verify_prefix(s)

if __name__ == '__main__': unittest.main()
