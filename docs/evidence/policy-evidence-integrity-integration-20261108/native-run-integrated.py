import hashlib,json,os,subprocess
from pathlib import Path
p=Path("/Users/numi/policy-evidence-integrity-20261108");r=Path("/Users/numi/numi-lab-reconciliation-20261108");install=Path("/Users/numi/numi-reconciliation-build-20261108");fixture=Path("/Users/numi/numi-autoload-franka-smoke-20261108/fixture");run=Path("/Users/numi/numi-autoload-franka-smoke-20261108/workspace/.numi/runs/window-train-smoke")
inputs=[r/"python/metalrobo/policy_selection.py",install/"bin/metalrobo_task_rollout",install/"shaders/MetalRobo.metallib",run/"incumbent.deployment.policypack",run/"candidate.deployment.policypack"]+[fixture/n for n in ("world.mrworld","task.taskpack","actuators.actuatorpack","sensors.sensorpack","reality.realitypack")]
def hashes():return {str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs}
inputs.append(install/"lib/libmetalrobo.dylib")
before=hashes();(p/"native-integrated-pretrial-sha256.json").write_text(json.dumps(before,indent=2,sort_keys=True)+"\n")
results=[]
for repeat in range(2):
 out=p/f"native-integrated-run-{repeat}";out.mkdir(exist_ok=False)
 cmd=["/opt/homebrew/bin/python3",str(inputs[0]),"--evaluator",str(inputs[1]),"--metallib",str(inputs[2]),"--incumbent",str(inputs[3]),"--candidate",str(inputs[4]),"--deployment",str(out/"deployment.pack"),"--evidence-directory",str(out/"evidence"),"--maximum-environments","1","--held-out-seed","4081019","--evaluation-steps","16","--","--world-pack",str(inputs[5]),"--task-pack",str(inputs[6]),"--robot-actuator-pack",str(inputs[7]),"--sensor-program-pack",str(inputs[8]),"--reality-program-pack",str(inputs[9]),"--envs","1","--steps","16"]
 env={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","NUMANX_"))}
 cp=subprocess.run(cmd,capture_output=True,text=True,timeout=180,env=env,cwd=out)
 (out/"stdout.txt").write_text(cp.stdout);(out/"stderr.txt").write_text(cp.stderr)
 row={"command":cmd,"returncode":cp.returncode,"inputs_unchanged":hashes()==before}
 if cp.returncode==0:
  d=json.loads(cp.stdout);pack=inputs[4] if d["selected"]=="candidate" else inputs[3];row.update(selected=d["selected"],deployment_matches=out.joinpath("deployment.pack").read_bytes()==pack.read_bytes(),regressions=d.get("regressions"),evidence=[])
  for name in ("incumbent","candidate"):
   a=json.loads((out/"evidence"/(name+".evidence.json")).read_text());row["evidence"].append({k:a.get(k) for k in ("task","world_source","termination_count","termination_reason_counts","termination_count_by_environment","physics_error_count_by_environment","failed_environment_steps","mean_task_reward","mean_reward","mean_tracking_score","clean_horizon_environment_rate")})
 results.append(row);(p/"native-integrated-results.json").write_text(json.dumps(results,indent=2,sort_keys=True)+"\n")
 print("repeat",repeat,"return",cp.returncode,"selected",row.get("selected"),"deployment matches",row.get("deployment_matches"),flush=True)
