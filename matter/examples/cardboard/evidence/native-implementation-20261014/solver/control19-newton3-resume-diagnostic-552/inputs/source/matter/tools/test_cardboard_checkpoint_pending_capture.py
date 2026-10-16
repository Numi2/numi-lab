import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cardboard_solver_environment as env


class PendingCheckpointCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.case = Path(self.temporary.name) / 'case'
        self.case.mkdir()
        self.key = 'NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH'
        self.prefixes = env.create_diagnostic_capture_prefixes(self.case, [self.key])
        self.contract = env.make_diagnostic_contract(
            {'NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR': '20:0:63:1:0:0'},
            self.case, self.prefixes)
        self.output = self.case / 'output'
        self.output.mkdir()
        self.checkpoint = self.output / 'accepted.nmcb'
        self.checkpoint.write_bytes(b'opaque native checkpoint fixture')
        digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.metadata = self.output / 'accepted.resume'
        self.metadata.write_text('schema=numi.cardboard.accepted-coupon-resume.v2\n'
            'accepted_control_index=19\nnext_control_index=20\nboth_environments_accepted=true\n'
            f'checkpoint_sha256={digest}\ncheckpoint_bytes={self.checkpoint.stat().st_size}\n')
        self.result = dict(schema='numi.cardboard.probe-result.v1',
            status='accepted_checkpoint_prefix_saved', accepted_steps=20, physical_validation=False)
        self.prefix = dict(schema='numi.cardboard.coupon-checkpoint-prefix.v1',
            status='accepted_checkpoint_prefix_saved', accepted_control_index=19, next_control_index=20,
            checkpoint_file='accepted.nmcb', checkpoint_sha256=digest,
            metadata_file='accepted.resume', metadata_sha256=hashlib.sha256(self.metadata.read_bytes()).hexdigest(),
            full_protocol_complete=False, trajectory_parity_established=False, physical_validation=False)
        self.write_reports()
        self.argv = ['probe', '--coupon-checkpoint-after-control', '19']

    def tearDown(self):
        self.temporary.cleanup()

    def write_reports(self):
        for name, report in [('result.json', self.result), ('coupon-checkpoint-prefix.json', self.prefix)]:
            (self.output / name).write_text(json.dumps(report))

    def validate(self, **changes):
        arguments = dict(native_argv=self.argv, native_exit_code=0, native_timed_out=False, effective_contract=self.contract)
        arguments.update(changes)
        return env.validate_diagnostic_outputs(self.case, self.prefixes, **arguments)

    def test_future_checkpoint_is_diagnostic_pending_not_capture_success(self):
        result = self.validate()
        self.assertEqual(result['status'], 'validated_with_pending_checkpoint_targets')
        self.assertEqual(result['files'], [])
        pending = result['pending_captures'][0]
        self.assertEqual((pending['accepted_control'], pending['target_control']), (19,20))
        self.assertFalse(pending['positive_qualification_allowed'])
        self.assertEqual(pending['checkpoint_sha256'], self.prefix['checkpoint_sha256'])
        with self.assertRaisesRegex(ValueError, 'no files'):
            env.validate_diagnostic_outputs(self.case, self.prefixes)

    def test_zero_exit_timeout_cannot_certify_pending_checkpoint(self):
        with self.assertRaisesRegex(ValueError, 'timed-out native process'):
            self.validate(native_exit_code=0, native_timed_out=True)

    def test_operator_fd_future_control_supported(self):
        key = 'NM_FEM_OPERATOR_FD_CAPTURE_PATH'
        Path(self.prefixes[self.key]).parent.rmdir()
        (self.case/'diagnostics').rmdir()
        prefixes = env.create_diagnostic_capture_prefixes(self.case, [key])
        contract = env.make_diagnostic_contract({'NM_FEM_OPERATOR_FD_SELECTOR':'20:63'}, self.case, prefixes)
        result = env.validate_diagnostic_outputs(self.case, prefixes, native_argv=self.argv,
            native_exit_code=0, effective_contract=contract)
        self.assertEqual(result['pending_captures'][0]['selector'], '20:63')

    def test_failed_native_or_missing_explicit_checkpoint_rejected(self):
        for changes in [dict(native_exit_code=2),dict(native_exit_code=-15),dict(native_exit_code=None),dict(native_timed_out=True),
                        dict(native_argv=['probe']),dict(native_argv=self.argv+self.argv[1:]),
                        dict(effective_contract=None)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.validate(**changes)

    def test_normal_result_and_false_completion_claim_rejected(self):
        for status in ['failed','completed','resumed_diagnostic_segment_reached_schedule_end']:
            self.result['status'] = status
            self.write_reports()
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.validate()
        self.result['status'] = 'accepted_checkpoint_prefix_saved'
        for key in ['full_protocol_complete','trajectory_parity_established','physical_validation']:
            self.prefix[key] = True
            self.write_reports()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()
            self.prefix[key] = False

    def test_already_visited_malformed_and_different_case_contract_rejected(self):
        for selector in ['19:0:63:1:0:0','0:0:63:1:0:0','all','20:0:63', '4294967296:0:63:1:0:0']:
            contract = env.make_diagnostic_contract(
                {'NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR':selector},self.case,self.prefixes)
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                self.validate(effective_contract=contract)
        other = self.case.parent / 'other'
        other.mkdir()
        prefixes = env.create_diagnostic_capture_prefixes(other,[self.key])
        contract = env.make_diagnostic_contract(
            {'NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR':'20:0:63:1:0:0'},other,prefixes)
        with self.assertRaisesRegex(ValueError, 'another case'):
            self.validate(effective_contract=contract)

    def test_checkpoint_metadata_and_path_tampering_rejected(self):
        self.checkpoint.write_bytes(b'modified')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            self.validate()
        self.checkpoint.write_bytes(b'opaque native checkpoint fixture')
        self.metadata.write_text(self.metadata.read_text()+'both_environments_accepted=true\n')
        self.prefix['metadata_sha256'] = hashlib.sha256(self.metadata.read_bytes()).hexdigest()
        self.write_reports()
        with self.assertRaisesRegex(ValueError,'duplicate'):
            self.validate()
        self.prefix['metadata_file'] = '../outside.resume'
        self.write_reports()
        with self.assertRaisesRegex(ValueError,'escaped'):
            self.validate()

    def test_nonempty_capture_still_validated_and_bad_entries_rejected(self):
        directory = Path(self.prefixes[self.key]).parent
        file = directory/'capture.json'
        file.write_text('{}')
        result = self.validate(native_exit_code=2)
        self.assertEqual(result['status'],'validated')
        self.assertEqual(result['pending_captures'],[])
        result = self.validate(native_exit_code=0, native_timed_out=True)
        self.assertEqual(result['status'],'validated')
        self.assertEqual(result['pending_captures'],[])
        file.unlink()
        outside = self.case.parent/'outside'
        outside.write_text('payload')
        file.symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'not a new regular'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
