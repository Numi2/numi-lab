import hashlib,json,pathlib,subprocess,time
repo=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019")
d33=repo/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/wholebody-attachment-source-rebind-033-final091-regenerated-deltas-001"
d34=repo/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/wholebody-attachment-source-rebind-034-main-f768-compatible-001"
base=pathlib.Path("/private/tmp/numi-main-f768-d34-base")
oldbuild=d33/"compile-033-attempt-001"; newbuild=d34/"build-main-f768-002"; newbuild.mkdir(parents=True,exist_ok=False)
oldcand=d33/"source/apps"; newoverlay=d34/"source-overlay/apps"
oldankle=repo/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/ankle-main-app-integration-001/source"
oldc=json.loads((oldbuild/"compile-command.json").read_text()); newc=[]
for x0 in oldc:
 x=str(x0)
 if x.startswith(str(oldbuild)): x=x.replace(str(oldbuild),str(newbuild))
 elif x.startswith(str(oldcand)): x=x.replace(str(oldcand),str(newoverlay))
 elif x.startswith("-I"+str(oldankle/"include")): x="-I"+str(base/"include")
 elif x.startswith("-I"+str(oldankle/"matter/include")): x="-I"+str(base/"matter/include")
 elif x.startswith("-I"+str(oldankle/"apps")): x="-I"+str(base/"apps")
 elif x==str(oldankle/"include"): x=str(base/"include")
 elif x==str(oldankle/"matter/include"): x=str(base/"matter/include")
 elif x==str(oldankle/"apps"): x=str(base/"apps")
 newc.append(x)
(newbuild/"compile-command.json").write_text(json.dumps(newc,indent=2)+"\n")
(newbuild/"compile-cwd.json").write_text(json.dumps({"cwd":str(repo)},indent=2)+"\n")
start=time.monotonic(); run=subprocess.run(newc,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.PIPE); elapsed=time.monotonic()-start
(newbuild/"compile.stdout.log").write_bytes(run.stdout); (newbuild/"compile.stderr.log").write_bytes(run.stderr)
(newbuild/"compile-result.json").write_text(json.dumps({"command_sha256":hashlib.sha256(json.dumps(newc,separators=(",",":")).encode()).hexdigest(),"elapsed_s":elapsed,"returncode":run.returncode,"stdout_sha256":hashlib.sha256(run.stdout).hexdigest(),"stderr_sha256":hashlib.sha256(run.stderr).hexdigest()},indent=2)+"\n")
print("compile",run.returncode,"elapsed",round(elapsed,3),"s")
if run.returncode: print(run.stderr.decode(errors="replace")); raise SystemExit(run.returncode)
oldl=json.loads((oldbuild/"link-command.json").read_text()); newl=[]
for x0 in oldl:
 x=str(x0)
 if x.startswith(str(oldbuild)): x=x.replace(str(oldbuild),str(newbuild))
 newl.append(x)
(newbuild/"link-command.json").write_text(json.dumps(newl,indent=2)+"\n")
start=time.monotonic(); run=subprocess.run(newl,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.PIPE); elapsed=time.monotonic()-start
(newbuild/"link.stdout.log").write_bytes(run.stdout); (newbuild/"link.stderr.log").write_bytes(run.stderr)
(newbuild/"link-result.json").write_text(json.dumps({"command_sha256":hashlib.sha256(json.dumps(newl,separators=(",",":")).encode()).hexdigest(),"elapsed_s":elapsed,"returncode":run.returncode,"stdout_sha256":hashlib.sha256(run.stdout).hexdigest(),"stderr_sha256":hashlib.sha256(run.stderr).hexdigest()},indent=2)+"\n")
print("link",run.returncode,"elapsed",round(elapsed,3),"s")
if run.returncode: print(run.stderr.decode(errors="replace")); raise SystemExit(run.returncode)
binsha=hashlib.sha256((newbuild/"numi-human-native").read_bytes()).hexdigest(); (newbuild/"binary.sha256").write_text(binsha+"  numi-human-native\n"); print("binary_sha256",binsha)
