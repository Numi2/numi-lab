import contextlib
import importlib.util
import io
import json
import unittest
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
E=Path("/Users/n/numi-human-resting-evidence-20261005")
SCENE=E/"final-native-scene-preflight-936/skin-927-lung-1078-viewer-018-v015-attempt1"
spec=importlib.util.spec_from_file_location("recorder_entrypoint_v017",ROOT/"record_treatment_program_probe.py")
recorder=importlib.util.module_from_spec(spec);spec.loader.exec_module(recorder)
class ActualRecorderEntrypointTests(unittest.TestCase):
    @unittest.skipUnless((SCENE/"native-run/run-metadata.json").is_file(),"requires retained completed1078 native scene")
    def test_plan_only_entrypoint_accepts_actual_source_inventory_without_creating_output(self):
        args=json.loads((SCENE/"treatment-probe-launch-argv.json").read_text())[2:]
        args.remove("--execute")
        output=SCENE/"native-run"/recorder.PROGRAM_PROBE_NAME
        if output.exists():
            self.skipTest("probe already recorded; prelaunch entrypoint evidence retained in test-suite-v017.log")
        self.assertFalse(output.exists())
        # Match the native owner's pinned Python: platform.platform() is SDK-sensitive
        # on macOS and must be reported by the same interpreter as the control.
        result=subprocess.run([str(recorder.load_readiness().PY39),
                               str(ROOT/"record_treatment_program_probe.py")]+args,
                              capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        text=result.stdout
        self.assertIn('"expected_steps": 10000',text)
        self.assertIn("Plan only: no directory, receipt, or native process was created.",text)
        self.assertFalse(output.exists())
if __name__=="__main__":unittest.main()
