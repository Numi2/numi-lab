from pathlib import Path
import json,subprocess,unittest
D=Path('/Users/numi/Work/numi-human-connected-surface-069')
ENTRY=D/'D/D/distance-slack-scaling-523/run_distance_slack_527.py'
LAUNCH=D/'D/finite-triangle-fit-511/weighted-surface-003-launch/argv.json'
RECEIPT=D/'D/D/distance-slack-scaling-523/entry-cli-preflight-527.json'
class EntryCliPreflight527(unittest.TestCase):
 def test_matched_fit_cli_installs_hook_before_485_without_fit(self):
  argv=json.loads(LAUNCH.read_text());argv[1]=str(ENTRY)
  for flag in ('--anchor-compatible-finite-triangle-513','--anchor-compatible-finite-triangle-all-pairs-513','--absolute-deficit-surface-weights-520'):
   self.assertEqual(argv.count(flag),1,flag)
  if '--distance-slack-scaling-523' not in argv:argv.append('--distance-slack-scaling-523')
  argv.append('--preflight-only-527')
  run=subprocess.run(argv,capture_output=True,text=True,timeout=30,check=False)
  self.assertEqual(run.returncode,0,run.stdout+'\n'+run.stderr)
  report=json.loads(run.stdout.strip().splitlines()[-1])
  self.assertEqual(report['status'],'CLI_PREFLIGHT_PASSED_NO_FIT')
  self.assertTrue(report['hook_attachment']['attached'])
  self.assertTrue(report['actual_loader_chain_preflight']['passed'])
  self.assertTrue(report['actual_loader_chain_preflight']['loaded470_installer_hooked'])
  self.assertFalse(report['fit_started']);self.assertFalse(report['geometry_started'])
  RECEIPT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':unittest.main()
