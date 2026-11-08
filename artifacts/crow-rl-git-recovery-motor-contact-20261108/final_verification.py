from pathlib import Path
import hashlib,json,subprocess,sys
R=Path.cwd();G=R/"artifacts/crow-rl-git-recovery-motor-contact-20261108";A=R/"artifacts/crow-rl-motor-contact-20261108"
sys.path.insert(0,str(R/"tools"));from crow_evidence_archive import tree,Objects,content
head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip();remote=subprocess.check_output(["git","ls-remote","origin","refs/heads/agent/american-crow-rl-20261107"],text=True).split()[0];assert head==remote
entries=tree()
with Objects() as objects:
 def committed(path):return objects.read(entries[str(path.relative_to(R))][1])
 plan=json.loads(committed(G/"plan.json"));raw=objects.read(entries[plan["manifest"]][1]);assert hashlib.sha256(raw).hexdigest()==plan["manifest_sha256"];m=json.loads(raw);unique=set()
 for f in m["files"]:
  key=f["sha256"],f["bytes"]
  if key not in unique:
   data=b"".join(content(f,entries,objects));assert len(data)==f["bytes"] and hashlib.sha256(data).hexdigest()==f["sha256"];unique.add(key)
  assert not (R/f["path"]).exists()
 precision=json.loads(committed(A/"precision/plan.json"));identity=precision["source_and_runtime_sha256"];current={}
 for source in ["CMakeLists.txt","apps/motor_contact_probe.cpp","src/metal/MetalWorld.mm"]:
  h=hashlib.sha256((R/source).read_bytes()).hexdigest();assert h==identity[source];current[source]=h
 for current_path,archived_path in [("build/lib/libmetalrobo.dylib",".numi/retained-runtime/motor-contact-factor-fix-20261108/libmetalrobo.dylib"),("build/shaders/MetalRobo.metallib",".numi/retained-runtime/motor-contact-factor-fix-20261108/metalrobo/MetalRobo.metallib"),("build/bin/metalrobo_motor_contact_probe",".numi/retained-runtime/motor-contact-factor-fix-20261108/probe-precision")]:
  h=hashlib.sha256((R/current_path).read_bytes()).hexdigest();assert h==identity[archived_path];current[current_path]=h
policy=R/"environments/american-crow/action-holds-v1/policies/deployment97.policypack";policy_sha=hashlib.sha256(policy.read_bytes()).hexdigest();assert policy_sha=="6a009417eac3e04831c95f46c24e78f943e825892124b972a58b5ee9eeed0389"
for name in ["build/crow-motor-contact-build-01","build/crow-motor-contact-git-recovery-01",".numi/retained-runtime/motor-contact-20261108",".numi/retained-runtime/motor-contact-factor-fix-20261108"]:assert not (R/name).exists()
processes=subprocess.check_output(["ps","-axo","pid=,command="],text=True);workers=[]
for line in processes.splitlines():
 parts=line.strip().split(None,1)
 if len(parts)==2 and parts[1].startswith((str(R/"build")+"/",str(R/".numi/retained-runtime")+"/")):workers.append(line)
assert not workers,workers
handles=subprocess.run(["/usr/sbin/lsof","-Fpn","+D",str(R/"build")],capture_output=True,text=True,timeout=60);assert handles.returncode in (0,1) and not handles.stdout and not handles.stderr
free=subprocess.check_output(["df","-k","/System/Volumes/Data"],text=True);free_kib=int(free.splitlines()[-1].split()[3])
report=dict(verification_commit=head,remote_commit=remote,manifest_sha256=plan["manifest_sha256"],actual_git_files_verified=len(m["files"]),actual_git_unique_contents_verified=len(unique),source_and_current_runtime_sha256=current,protected_policy_sha256=policy_sha,original_raw_and_temporary_scopes_absent=True,own_native_workers=workers,build_open_handles=0,df=free,data_available_GiB=free_kib/1024**2,policy_or_profile_promoted=False,goal_complete=False)
(G/"final-verification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report))
