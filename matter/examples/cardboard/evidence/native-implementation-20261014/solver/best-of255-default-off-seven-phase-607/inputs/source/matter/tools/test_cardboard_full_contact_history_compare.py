import json
import shutil
import tempfile
import unittest
from pathlib import Path
import cardboard_full_contact_history_compare as compare
import test_cardboard_diagnostic_history_compare as fixtures


def write(path,obj):path.write_text(json.dumps(obj,sort_keys=True)+'\n')


class FullContactHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.left=self.root/'diagnostic';self.right=self.root/'control'
        maker=fixtures.DiagnosticHistoryComparatorTests()
        for case,enabled in ((self.left,True),(self.right,False)):
            maker._make_attempt(case,enabled)
            shutil.rmtree(case/'output');out=case/'diagnostic-state';out.mkdir()
            p=json.loads((case/'prediction.json').read_text())
            p['argv'][-2:]=['--diagnostic-state-output',str(out)]
            p['environment']={'NM_FEM_CONTACT_TRACE_ROOT':'0','NM_FEM_CONTACT_TRACE_ITERATION':'20',
                'NM_FEM_NEWTON_TRACE_ROOT':'0','NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT':'0'} if enabled else {}
            write(case/'prediction.json',p)
            state=dict(available=True,control_step=0,physics_substep=0,statuses=[dict(code=0)],solver_certificates=[],
                vector_counts={k:0 for k in compare.PERSISTENT_VECTORS},
                accepted_state_vector_sha256={k:'0'*64 for k in compare.PERSISTENT_VECTORS})
            after=json.loads(json.dumps(state));after['statuses']=[dict(code=10)]
            row=dict(schema='numi.cardboard.full_contact_state_attempt.v1',attempt_index=0,requested_control_step=0,
                completion_stage='snapshot_read',before_file='transaction-000000-before.bin',
                after_file='transaction-000000-after.bin',after_present=True,accepted=False,before=state,after=after)
            write(out/'transaction-000000.json',row)
            (out/'transaction-000000-before.bin').write_bytes(b'raw before accepted state')
            (out/'transaction-000000-after.bin').write_bytes(b'raw after with failed certificate')

    def tearDown(self):self.tmp.cleanup()

    def mutate(self,case,change):
        path=case/'diagnostic-state/transaction-000000.json';r=json.loads(path.read_text());change(r);write(path,r)

    def test_failed_raw_payloads_have_parity_and_persistent_rollback(self):
        r=compare.compare(self.left,self.right)
        self.assertTrue(r['production_outputs_byte_identical']);self.assertEqual(r['output_file_count'],3)
        self.assertFalse(r['strict_solver_qualification'])
        self.assertEqual(r['attempts']['control']['accepted_transactions'],0)
        self.assertEqual(r['attempts']['control']['rollback'][0]['rollback'],'persistent_scope_identical')

    def enable_true_residual(self, replacement='0', emit=True):
        p=json.loads((self.left/'prediction.json').read_text())
        p['environment'].update(NM_FEM_TRUE_RESIDUAL_ROOT='0', NM_FEM_TRUE_RESIDUAL_RESTART=replacement)
        write(self.left/'prediction.json',p)
        if emit:
            (self.left/'stderr.log').write_text('fem_true_residual_cycle='+json.dumps(dict(schema=2,root=0,restart_replacement=False,operator_scope='full_fem_mass_bulk_cohesive_and_kinematic_rigid_self_contact',environments=[dict(operator_status_code=0,solver_status_code=0,true_l2=1e-7,arnoldi_estimate=9e-8,true_max_abs=1e-8,rhs_l2=1e-3)]))+'\n')

    def test_read_only_complete_true_residual_is_admitted(self):
        self.enable_true_residual()
        self.assertTrue(compare.compare(self.left,self.right)['production_outputs_byte_identical'])

    def test_residual_replacement_is_not_diagnostic_parity(self):
        self.enable_true_residual('1')
        with self.assertRaisesRegex(compare.common.ComparisonError,'numerical algorithm difference'):
            compare.compare(self.left,self.right)

    def test_missing_true_residual_capture_is_inconclusive(self):
        self.enable_true_residual(emit=False)
        with self.assertRaisesRegex(compare.common.ComparisonError,'not observed'):
            compare.compare(self.left,self.right)

    def test_invalid_true_residual_metrics_are_rejected(self):
        for invalid in (None, float('nan'), float('inf'), -1, True):
            self.enable_true_residual()
            path=self.left/'stderr.log'
            row=json.loads(path.read_text().split('=',1)[1])
            row['environments'][0]['true_l2']=invalid
            path.write_text('fem_true_residual_cycle='+json.dumps(row)+'\n')
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(compare.common.ComparisonError,'invalid metric'):
                    compare.compare(self.left,self.right)

    def test_failed_production_status_does_not_admit_residual_metric(self):
        self.enable_true_residual()
        path=self.left/'stderr.log';row=json.loads(path.read_text().split('=',1)[1])
        row['environments'][0]['solver_status_code']=10
        path.write_text('fem_true_residual_cycle='+json.dumps(row)+'\n')
        with self.assertRaisesRegex(compare.common.ComparisonError,'production solver failed'):
            compare.compare(self.left,self.right)

    def test_valid_metrics_remain_distinct_from_failed_production_rows(self):
        self.enable_true_residual()
        path=self.left/'stderr.log';row=json.loads(path.read_text().split('=',1)[1])
        row['environments'][0].update(solver_status_code=6,true_l2=None)
        with path.open('a') as stream:
            stream.write('fem_true_residual_cycle='+json.dumps(row)+'\n')
        report=compare.compare(self.left,self.right)
        self.assertTrue(report['production_outputs_byte_identical'])
        self.assertEqual(report['true_residual_instrument']['healthy_production_rows'],1)
        self.assertEqual(report['true_residual_instrument']['failed_production_environment_rows_excluded'],1)

    def test_raw_after_mismatch_is_reported(self):
        (self.right/'diagnostic-state/transaction-000000-after.bin').write_bytes(b'changed')
        r=compare.compare(self.left,self.right)
        self.assertFalse(r['production_outputs_byte_identical'])
        self.assertIn('transaction-000000-after.bin',r['differing_files'])

    def test_identical_bad_rollback_does_not_become_solver_qualification(self):
        for case in (self.left,self.right):
            self.mutate(case,lambda r:r['after']['accepted_state_vector_sha256'].update(femMaterialState='1'*64))
        r=compare.compare(self.left,self.right)
        self.assertTrue(r['production_outputs_byte_identical'])
        self.assertEqual(r['attempts']['control']['rollback'][0]['different_vectors'],['femMaterialState'])
        self.assertEqual(r['attempts']['control']['rollback'][0]['rollback'],'persistent_scope_changed')
        self.assertFalse(r['strict_solver_qualification'])

    def test_cursor_change_is_reported(self):
        self.mutate(self.right,lambda r:r['after'].update(control_step=1))
        r=compare.compare(self.left,self.right)
        self.assertEqual(r['attempts']['control']['rollback'][0]['changed_cursors'],['control_step'])

    def test_missing_persistent_hash_scope_is_rejected(self):
        self.mutate(self.right,lambda r:r['after']['accepted_state_vector_sha256'].pop('femNodes'))
        with self.assertRaisesRegex(compare.common.ComparisonError,'scope is incomplete'):
            compare.compare(self.left,self.right)

    def test_failed_status_cannot_be_marked_accepted(self):
        self.mutate(self.right,lambda r:r.update(accepted=True))
        with self.assertRaisesRegex(compare.common.ComparisonError,'failed/missing production status'):
            compare.compare(self.left,self.right)

    def test_missing_after_state_is_inconclusive(self):
        for case in (self.left,self.right):
            self.mutate(case,lambda r:r.update(after_present=False))
            (case/'diagnostic-state/transaction-000000-after.bin').unlink()
        r=compare.compare(self.left,self.right)
        self.assertEqual(r['attempts']['control']['rollback'][0]['rollback'],'inconclusive_missing_after_state')

    def test_tampered_sealed_input_is_rejected(self):
        (self.right/'inputs/runtime/probe').write_bytes(b'tampered')
        with self.assertRaisesRegex(compare.common.ComparisonError,'sealed input hash mismatch'):
            compare.compare(self.left,self.right)

    def test_physics_argument_change_is_rejected(self):
        path=self.right/'prediction.json';p=json.loads(path.read_text());p['argv'][2]='anothergrade';write(path,p)
        with self.assertRaisesRegex(compare.common.ComparisonError,'physical argv'):
            compare.compare(self.left,self.right)

    def test_timeout_is_not_parity_admitted(self):
        path=self.right/'receipt.json';p=json.loads(path.read_text());p['timed_out']=True;write(path,p)
        with self.assertRaisesRegex(compare.common.ComparisonError,'timed out/changed'):
            compare.compare(self.left,self.right)

    def test_unlisted_environment_change_is_rejected(self):
        path=self.right/'prediction.json';p=json.loads(path.read_text());p['environment']['UNKNOWN_SOLVER_SWITCH']='1';write(path,p)
        with self.assertRaisesRegex(compare.common.ComparisonError,'physical environment'):
            compare.compare(self.left,self.right)


if __name__=='__main__':unittest.main()
