"""CPU-only synthetic cross-language contract exercise; no physical result."""
import json
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, '/Users/numi/numi-lab-catalog-clean-installed-retry-20261108/share/numi/python')
from metalrobo.policy_selection import _evaluate

binary = sys.argv[1]
with tempfile.TemporaryDirectory() as tmp:
    run = Path(tmp)
    selection = run / 'selection'
    selection.mkdir()
    (selection / 'selection.json').write_text('{"selected":"incumbent"}')
    policy, shader, world, urdf, evaluator = [run / x for x in ('deployment.policypack', 'shader', 'world', 'robot.urdf', 'evaluator')]
    for p, value in ((policy, 'policy'), (shader, 'shader'), (world, 'world-a'), (urdf, 'urdf')):
        p.write_text(value)
    trace = selection / 'incumbent.state.tsv'
    record = {'world_source':'bundled_g1', 'failed_environment_steps':0,
              'height_or_tilt_termination_count':0, 'mean_tilt':0.2,
              'clean_horizon_environment_rate':0.75, 'minimum_root_height_by_environment':[0.6]}
    evaluator.write_text("#!/bin/sh\nprintf '# step nq=1 scene_bodies=0 scene_stride=13 timestep=0.02 environment=0\\n1\\t0\\n' > " + shlex.quote(str(trace)) + "\nprintf '%s\\n' " + shlex.quote(json.dumps(record)) + "\n")
    evaluator.chmod(0o755)
    args = ['--world-pack', str(world), '--policy-pack', str(policy), '--metallib', str(shader), '--state-trace', str(trace), '--steps', '1']
    evidence = selection / 'incumbent.evidence.json'
    def check(name, expected):
        observed = subprocess.check_output([binary, str(run)], text=True).strip()
        print(name + ': ' + observed)
        assert observed == expected
    passed = 'guided held-out balance screen passed|load=true'
    unknown = 'guided held-out balance status unknown|load=false'
    _evaluate(evaluator, args, evidence)
    check('producer metadata', passed)
    world.write_text('world-b')
    check('same path replaced', unknown)
    world.unlink()
    check('missing input', unknown)
    world.write_text('world-a')
    _evaluate(evaluator, args + ['--urdf', str(urdf)], evidence)
    check('unsupported dependency closure', unknown)
