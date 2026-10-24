from pathlib import Path
import json,subprocess
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019");E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/"native-candidate020-mechanical-source-024";N=E/"native-candidate020-bone-witness-028"
N.mkdir(exist_ok=False)
cmd=json.loads((D/"native-1ms-command.json").read_text());cmd[cmd.index("--base-manifest")+1]=str(D/"native-1ms-001/launch-manifest.snapshot.json");cmd[cmd.index("--run-dir")+1]=str(N/"source-preflight-001");cmd[cmd.index("--mode")+1]="source-preflight"
cmd+=["--env","NUMI_HUMAN_INITIAL_SKIN_BONE_WITNESS_CSV="+str(N/"not-used-remapped.csv")]
(N/"command.json").write_text(json.dumps(cmd,indent=2))
p=subprocess.run(cmd,cwd=R);(N/"runner-exit.json").write_text(json.dumps(dict(returncode=p.returncode)));raise SystemExit(p.returncode)
