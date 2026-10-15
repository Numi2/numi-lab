import importlib.util
from pathlib import Path
import unittest

SCRIPT=Path(__file__).with_name('audit_capture_observer_parity.py')
spec=importlib.util.spec_from_file_location('observer_parity',SCRIPT)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class EnvTests(unittest.TestCase):
    def test_exact_two_capture_keys_removed_and_runpaths_normalized(self):
        on={'NM_FEM_BASE_CGLS_CAPTURE_NEWTON':'48','NM_FEM_BASE_CGLS_CAPTURE_PATH':'/r/on/base.bin',
            'NM_FEM_BASE_CGLS_ITERATIONS':'64','NM_FEM_STATUS5_CAPTURE_PATH':'/r/on/first-status5'}
        off={'NM_FEM_BASE_CGLS_ITERATIONS':'64','NM_FEM_STATUS5_CAPTURE_PATH':'/r/off/first-status5'}
        pred_on={'environment':on}; pred_off={'environment':off}
        got=m.validate_only_capture_env_removed(on,off,pred_on,pred_off,Path('/r/on'),Path('/r/off'))
        self.assertTrue(got['removed_keys_exact'])
        with self.assertRaisesRegex(ValueError,'exactly'):
            m.validate_only_capture_env_removed(on,{**off,'EXTRA':'1'},pred_on,{'environment':{**off,'EXTRA':'1'}},Path('/r/on'),Path('/r/off'))
        with self.assertRaisesRegex(ValueError,'non-capture wrapper environment differs'):
            m.validate_only_capture_env_removed(on,{**off,'NM_FEM_BASE_CGLS_ITERATIONS':'32'},pred_on,{'environment':{**off,'NM_FEM_BASE_CGLS_ITERATIONS':'32'}},Path('/r/on'),Path('/r/off'))

    def test_probe_only_output_path_normalizes(self):
        a=['--steps','12','--output','/r/on/output','--dt','.0001']
        b=['--steps','12','--output','/r/off/output','--dt','.0001']
        self.assertEqual(m.normalize_probe_args(a,Path('/r/on')),m.normalize_probe_args(b,Path('/r/off')))
        b[1]='13'
        with self.assertRaises(ValueError):
            if m.normalize_probe_args(a,Path('/r/on')) != m.normalize_probe_args(b,Path('/r/off')):
                raise ValueError('physical probe arguments differ')

    def test_capture_and_timing_filtered_but_solver_records_remain(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'stderr'; root=Path(td)
            p.write_text('cardboard_step_timing={"x":1}\nfem_coupled_capture_skipped={"x":1}\nfem_base_cgls_direction={"x":1}\nfem_base_cgls_enabled={"x":1}\nfem_residual_armijo_iteration={"x":1}\nfem_true_residual_cycle={"x":1}\n')
            lines=m.filtered_stderr(p,root)
            self.assertEqual(lines,['fem_base_cgls_enabled={"x":1}','fem_residual_armijo_iteration={"x":1}','fem_true_residual_cycle={"x":1}'])

if __name__=='__main__': unittest.main()
