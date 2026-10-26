from pathlib import Path
import json, subprocess, unittest

D=Path('/Users/numi/Work/numi-human-connected-surface-069')
ENTRY=D/'D/D/absolute-deficit-surface-weighting-520/run_absolute_deficit_521.py'
LAUNCH=D/'D/finite-triangle-fit-511/weighted-surface-003-launch/argv.json'
RECEIPT=D/'D/D/absolute-deficit-surface-weighting-520/entry-cli-preflight-521.json'

class ActualEntryCliPreflight521(unittest.TestCase):
    def test_actual_entry_cli_with_matched_launch_arguments_without_fit(self):
        argv=json.loads(LAUNCH.read_text())
        argv[1]=str(ENTRY)
        for flag in ('--anchor-compatible-finite-triangle-513',
                     '--anchor-compatible-finite-triangle-all-pairs-513',
                     '--absolute-deficit-surface-weights-520'):
            self.assertEqual(argv.count(flag),1,flag)
        self.assertFalse(any(x=='--preflight-only-521' for x in argv))
        argv.append('--preflight-only-521')
        completed=subprocess.run(argv,capture_output=True,text=True,check=False,timeout=30)
        self.assertEqual(completed.returncode,0,completed.stdout+'\n'+completed.stderr)
        report=json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(report['status'],'CLI_PREFLIGHT_PASSED_NO_FIT')
        self.assertEqual(report['entry_521'],str(ENTRY))
        self.assertEqual(report['finite_triangle_helper'],str(D/'D/D/anchor-compatible-finite-triangle-513/anchor_finite_triangle_513.py'))
        self.assertTrue(report['all_pair_mode'])
        self.assertFalse(report['fit_started'])
        self.assertTrue(report['global_surface_clearance_certified'] is False)
        RECEIPT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')

if __name__=='__main__': unittest.main()
