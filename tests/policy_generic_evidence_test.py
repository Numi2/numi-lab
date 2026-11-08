#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class GenericEvidenceCLITest(unittest.TestCase):
    def test_selector_deployment_is_bound_to_complete_physical_evidence(self):
        selector = Path(__file__).resolve().parents[1] / "python/metalrobo/policy_selection.py"
        for mode in ("clean", "missing_total", "missing_failed", "mismatched_total", "physics_error", "missing_source", "missing_source_implicit", "wrong_source"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                incumbent, candidate = root / "incumbent.pack", root / "candidate.pack"
                world, shader = root / "world.pack", root / "shader"
                deployment, evidence, evaluator = root / "deployment.pack", root / "evidence", root / "evaluator.py"
                incumbent.write_bytes(b"incumbent"); candidate.write_bytes(b"candidate")
                world.write_bytes(b"world"); shader.write_bytes(b"shader")
                evaluator.write_text(
                    "#!" + sys.executable + "\n"
                    "import json,pathlib,sys\n"
                    f"mode={mode!r}\n"
                    "a=sys.argv[1:]; path=lambda x:pathlib.Path(a[a.index(x)+1])\n"
                    "candidate=path('--policy-pack').name=='candidate.pack'\n"
                    "path('--state-trace').write_text('# step nq=1 scene_bodies=0 scene_stride=13 timestep=0.02 environment=0\\n1\\t0\\n')\n"
                    "r={'task':'grasp','world_source':'world_pack','termination_count':0,'termination_count_by_environment':[0,0],'termination_reason_counts':{},'physics_error_count_by_environment':[0,0],'failed_environment_steps':0,'mean_reward':0.9 if candidate else 0.1,'mean_task_reward':0.9 if candidate else 0.1,'mean_tracking_score':0}\n"
                    "if mode in ('missing_source','missing_source_implicit'): r.pop('world_source')\n"
                    "if mode=='wrong_source': r['world_source']='urdf'\n"
                    "if candidate and mode=='missing_total': r.pop('termination_count'); r['termination_count_by_environment']=[1,0]\n"
                    "if candidate and mode=='missing_failed': r.pop('failed_environment_steps')\n"
                    "if candidate and mode=='mismatched_total': r['termination_count_by_environment']=[1,0]\n"
                    "if candidate and mode=='physics_error': r['physics_error_count_by_environment']=[0,1]\n"
                    "print(json.dumps(r))\n", encoding="utf-8")
                evaluator.chmod(0o755)
                command = [sys.executable, str(selector), "--evaluator", str(evaluator),
                           "--metallib", str(shader), "--incumbent", str(incumbent),
                           "--candidate", str(candidate), "--deployment", str(deployment),
                           "--evidence-directory", str(evidence), "--evaluation-steps", "1",
                           "--", "--task", "grasp", "--world-pack", str(world), "--envs", "2", "--steps", "1"]
                if mode == "missing_source_implicit":
                    index = command.index("--world-pack")
                    del command[index:index + 2]
                completed = subprocess.run(command, capture_output=True, text=True)
                decision = json.loads(completed.stdout)
                if mode == "clean":
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertEqual(decision["selected"], "candidate")
                    self.assertEqual(deployment.read_bytes(), b"candidate")
                else:
                    self.assertEqual(decision["selected"], "incumbent")
                    self.assertFalse(decision["candidate_advanced_deployment"])
                    self.assertEqual(deployment.read_bytes(), b"incumbent")
                    self.assertTrue(decision.get("regressions") or decision.get("selection_error"))
                self.assertEqual(candidate.read_bytes(), b"candidate")
                self.assertTrue((evidence / "selection.json").is_file())


if __name__ == "__main__":
    unittest.main()
