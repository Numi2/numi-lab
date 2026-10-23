#!/usr/bin/env python3
"""One-pose static native skin-registration export using the existing owner hook."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, signal, subprocess, time
from pathlib import Path

ROOT=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=ROOT/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
BINARY=Path("/tmp/numi-human-native-joints-clean-014")
BINARY_SHA="d6fb67d0bf16ffb6bed04a5887cf3916ab67e152d10006bb0eb0da47e1d5cd95"
RIGID=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/joint-functional-owner-audit-20261023/knee-projection-envelope-001/myosim-fullbody-core-reference.nhrigid')
RIGID_SHA='c10924a45efc1cfb3916bd22baabeeca491da5e412bbaad37b1328fe364db53d'
SKIN=E/'finger-profile-069i-left-ring-web-retreat/replacement-bilateral-hands038.nhskin'
SKIN_SHA='a848a41368aeb0dda8af579fe0ea417bbf7ec7db641033b5ba917302189fe1bc'
TEMPLATE=E/'native-pose-checks/candidate-registered-dump-001/result.json'
POSES={
 "construction-neutral": [],
 'shoulder-elevation-050': [(36,0.5),(74,0.5)],
 'shoulder-abduction-030': [(37,0.3),(75,0.3)],
 'elbow-flexion-080': [(39,0.8),(77,0.8)],
 'knee-flexion-025': [(106,0.25),(120,0.25)],
 'knee-flexion-005': [(106,0.05),(120,0.05)],
 'ankle-dorsiflexion-020': [(109,0.2),(123,0.2)],
 'ankle-plantarflexion-020': [(109,-0.2),(123,-0.2)],
 'mtp-flexion-020': [(111,0.2),(125,0.2)],
 'mtp-extension-020': [(111,-0.2),(125,-0.2)],
}

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def make_argv(output:Path, qrows:list[tuple[int,float]], binary:Path):
 prior=json.loads(TEMPLATE.read_text())
 a=list(prior['argv'])
 a[0]=str(binary);a[1]=str(RIGID);a[4]=str(output)
 a[a.index('--skin-payload')+1]=str(SKIN)
 if '--muscle-activation' in a:a[a.index('--muscle-activation')+1]='0.0'
 a=[x for x in a if x!='--resting-rigid-hands']
 cleaned=[];i=0
 while i<len(a):
  if a[i]=='--pose-q': i+=3;continue
  cleaned.append(a[i]);i+=1
 a=cleaned
 if '--mechanics-only' not in a:a.append('--mechanics-only')
 for q,v in qrows:a.extend(['--pose-q',str(q),format(v,'.8f')])
 assert '--resting-rigid-hands' not in a and '--root-pose' not in a
 assert a.count('--pose-q')==len(qrows)
 return a,prior

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pose',choices=sorted(POSES),required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--binary',type=Path,default=BINARY);ap.add_argument('--binary-sha256',default=BINARY_SHA);ap.add_argument('--timeout-seconds',type=float,default=180.0);args=ap.parse_args()
 out=args.output if args.output.is_absolute() else ROOT/args.output
 if not out.resolve().is_relative_to(E.resolve()):raise SystemExit('output must be inside the evidence directory')
 if out.exists():raise SystemExit(f'refusing to overwrite existing evidence: {out}')
 for p,h in ((RIGID,RIGID_SHA),(SKIN,SKIN_SHA)):
  if sha(p)!=h:raise SystemExit(f'input hash mismatch: {p}')
 out.mkdir(parents=True,exist_ok=False)
 csvp=out/'registered-skin.csv'
 posep=out/'registered-skin-poses.json'
 binary=args.binary.resolve()
 if sha(binary)!=args.binary_sha256:raise SystemExit(f'binary hash mismatch: {binary}')
 argv,prior=make_argv(out,POSES[args.pose],binary)
 # Keep only the runtime environment required to construct the existing
 # registered-skin owner. No ongoing drive, whole-body layer admission, or
 # unrelated organ/vascular FEM switches are enabled.
 env={k:v for k,v in os.environ.items() if not (k.startswith('NUMI_HUMAN_') or k.startswith('NM_FEM_HUMAN_'))}
 env.update({
  'NUMI_HUMAN_SKIN_REFERENCE_DUMP':str(csvp),
  'NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP':str(posep),
  'NUMI_HUMAN_RESTING_PASSIVE_FASCIA':'1',
  'NUMI_HUMAN_RESTING_SKIN_FAT_FEM':'1',
  'NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM':'0',
  'NUMI_HUMAN_RESTING_INFERRED_ACL':'1',
  'NUMI_HUMAN_RESTING_AORTIC_WALL':'0',
  'NUMI_HUMAN_RESTING_AORTIC_WALL_OUTER_CONTACT':'0',
  'NUMI_HUMAN_RESTING_PROSTATE_FEM':'0',
  'NUMI_HUMAN_RESTING_ORGAN_BATCH_FEM':'0',
  'NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE':'0',
  'NUMI_HUMAN_BLEEDING_ENABLED':'0',
  'NUMI_HUMAN_BLOOD_MPM_ENABLED':'0',
  # Existing native ropeDefaultPoseLaunch admission; the native owner adds a
  # fixed +0.35m root-z construction lift, then projects NHEQ.
  'NUMI_HUMAN_ROPE_ANCHOR':'0,0,0.8,0.8',
  'NUMI_HUMAN_ROPE_SITE':'-0.0545973483,0.1411030829,-0.00279765767',
 })
 inputs={}
 for i,x in enumerate(argv[1:]):
  p=Path(x)
  if p.is_file() and p.resolve()!=binary:inputs[str(p)]=sha(p)
 receipt={'schema':'numi.human.native-joint-skin-registration-dump.v1','runner_path':str(Path(__file__).resolve()),'runner_sha256':sha(Path(__file__)),'pose_name':args.pose,'requested_q':{str(k):v for k,v in POSES[args.pose]},'static_pose_only':True,'pose_owner':'existing --pose-q + NHEQ1 projection in resting construction path','ongoing_q_lock':False,'muscle_activation':0.0,'whole_body_skin_fat_admission_enabled':False,'construction_root_lift_m':0.35,'construction_root_transform_owner':'existing ropeDefaultPoseLaunch fixed +0.35m root-z from source root, followed by loaded NHEQ projection','common_world_frame':'all samples use the same source root transform and registered assembly coordinates; no per-sample world recentering','binary':str(binary),'binary_sha256':sha(binary),'candidate_rigid':str(RIGID),'candidate_rigid_sha256':sha(RIGID),'skin_payload':str(SKIN),'skin_payload_sha256':sha(SKIN),'input_hashes':inputs,'argv':argv,'working_directory':str(ROOT),'environment_overrides':{k:env[k] for k in env if k.startswith('NUMI_HUMAN_') or k.startswith('NM_FEM_HUMAN_')},'source_registration_hook':'NumiHumanSkinFEMIntegration.wholebody.hpp::makeRegisteredSkinMechanicalRegion; writes registered float32-derived source positions before whole-body consumer-row and layer-admission checks','paired_pose_snapshot':str(posep),'paired_pose_snapshot_owner':'same current bodies and skinRestBodies vectors passed to the CSV hook; existing writeSkinPoseSnapshot schema','boundary':'Paired native source registration and pose snapshots only; a later FEM initialization failure remains a failure, and no physics-step, replay, contact, or whole-body FEM qualification is claimed.'}
 (out/'invocation.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 t0=time.monotonic();rows=0;stable=0;last_size=-1;pose_stable=0;last_pose_size=-1;capture_ready_at=None;proc=None;termination_reason='native_process_exit'
 with (out/'stdout.log').open('wb') as so,(out/'stderr.log').open('wb') as se:
  proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=so,stderr=se)
  while time.monotonic()-t0<args.timeout_seconds:
   if csvp.exists():
    size=csvp.stat().st_size
    stable=stable+1 if size==last_size else 0;last_size=size
    if size>2_000_000 and stable>=2:
     with csvp.open(newline='') as f: rows=sum(1 for _ in csv.DictReader(f))
   if posep.exists():
    size=posep.stat().st_size
    pose_stable=pose_stable+1 if size==last_pose_size else 0;last_pose_size=size
   pose_valid=False
   if pose_stable>=2:
    try:
     snapshot=json.loads(posep.read_text())
     pose_valid=(snapshot.get('schema')=='numi.human.native-skin-pose-snapshot.v1' and snapshot.get('vertex_count')==61950 and snapshot.get('binding_count')==86 and len(snapshot.get('bodies',[]))==86)
    except (OSError,ValueError): pass
   if rows==61950 and pose_valid:
    if capture_ready_at is None: capture_ready_at=time.monotonic()
    elif time.monotonic()-capture_ready_at>=10.0:
     termination_reason='interrupted_after_complete_pre_admission_capture'
     break
   if proc.poll() is not None:break
   time.sleep(.5)
  requested_stop=False
  if proc.poll() is None:
   requested_stop=True
   if termination_reason=='native_process_exit':termination_reason='timeout'
   proc.send_signal(signal.SIGINT)
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
 if csvp.exists():
  with csvp.open(newline='') as f:rows=sum(1 for _ in csv.DictReader(f))
 csv_complete=rows==61950
 result={'schema':'numi.human.native-joint-skin-registration-dump-result.v1','pose_name':args.pose,'native_exit_code':proc.returncode if proc else None,'elapsed_seconds':time.monotonic()-t0,'timeout_seconds':args.timeout_seconds,'stop_requested_after_complete_pre_admission_capture':termination_reason=='interrupted_after_complete_pre_admission_capture','stop_requested_after_timeout':termination_reason=='timeout','termination_reason':termination_reason,'complete_61950_vertex_dump':csv_complete,'native_process_success':bool(proc and proc.returncode==0),'registered_rows':rows,'registered_csv_sha256':sha(csvp) if csvp.is_file() else None,'paired_pose_snapshot_exists':posep.is_file(),'paired_pose_snapshot_sha256':sha(posep) if posep.is_file() else None,'paired_pose_snapshot_valid':bool(posep.is_file() and json.loads(posep.read_text()).get('schema')=='numi.human.native-skin-pose-snapshot.v1' and json.loads(posep.read_text()).get('vertex_count')==61950 and json.loads(posep.read_text()).get('binding_count')==86),'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),'construction_pose_lines':[line for line in (out/'stdout.log').read_text(errors='replace').splitlines() if line.startswith('rope_initial_pose=') or line.startswith('rope_initial_equality_projection=')],'outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('result.json',)}}
 (out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps(result,indent=2),flush=True)
 return 0 if csv_complete and proc and proc.returncode==0 else 1
if __name__=='__main__':raise SystemExit(main())
