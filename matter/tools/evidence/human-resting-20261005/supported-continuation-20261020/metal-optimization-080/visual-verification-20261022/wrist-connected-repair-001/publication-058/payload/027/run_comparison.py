from pathlib import Path
import json,subprocess,hashlib
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019");E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/"native-candidate020-mechanical-source-024";W=E/"skin-material-jacobian-witness-154-zero-trial";N=E/"native-candidate020-weighted-support-027"
N.mkdir(exist_ok=False);base=json.loads((D/"native-1ms-001/launch-manifest.snapshot.json").read_text());a=base["argv"];out=[];i=0
while i<len(a):
 if a[i]=="--selected-tendon-control":i+=1;continue
 if a[i]=="--activated-source-muscle-index":i+=2;continue
 out.append(a[i]);i+=1
out[out.index("--muscle-activation")+1]="1.0";base["argv"]=out
base["scope"]="Existing weighted skin-bed support recruitment comparison, not matched-algorithm parity nor settled equilibrium"
(N/"base-manifest.json").write_text(json.dumps(base,indent=2))
policy=dict(existing_option="NUMI_HUMAN_STATIC_WEIGHTED_SKIN_SUPPORT=1",remove_selected_tendon_increment=True,remove_selected_muscle_subset=True,muscle_activation_semantics="Upper recruitment bound 1.0, not prescribed uniform activation; full source muscles recruit against registered skin-bed support witnesses",source_pose="unchanged identity004",physical_limits="unchanged",qualification="Fresh weighted support release comparison; not certified settled rest",baseline=str(D/"native-1ms-001/launch-manifest.snapshot.json"))
(N/"initialization-policy.json").write_text(json.dumps(policy,indent=2))
cmd=json.loads((D/"native-1ms-command.json").read_text());cmd[cmd.index("--base-manifest")+1]=str(N/"base-manifest.json");cmd[cmd.index("--run-dir")+1]=str(N/"run-001");cmd[cmd.index("--metallib")+1]=str(W/"build-004/candidate.metallib")
cmd+=["--source-root",str(N/"initialization-policy.json"),"--env","NUMI_HUMAN_STATIC_WEIGHTED_SKIN_SUPPORT=1","--env","NM_FEM_NEWTON_TRACE_ROOT=0"]
(N/"command.json").write_text(json.dumps(cmd,indent=2))
p=subprocess.run(cmd,cwd=R);(N/"runner-exit.json").write_text(json.dumps(dict(returncode=p.returncode)))
raise SystemExit(p.returncode)
