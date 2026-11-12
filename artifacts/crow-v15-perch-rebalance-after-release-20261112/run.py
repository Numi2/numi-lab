from pathlib import Path
import gzip,hashlib,json,os,subprocess,sys,threading
R=Path.cwd();A=R/"artifacts/crow-v15-perch-rebalance-after-release-20261112";B=R/"build/crow-v15-perch-rebalance-after-release-20261112";arm=sys.argv[1];assert arm in ("control","high");D=A/arm;D.mkdir(exist_ok=False)
probe=B/"probe";lib=R/"build/crow-v15-hip-roll-substep-observer-20261111/libmetalrobo.dylib";metal=R/"build/crow-v15-wide-contact-isolation-20261111/MetalRobo.metallib";h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(D/"inputs.json").write_text(json.dumps({"arm":arm,"probe_sha256":h(probe),"observer_host_sha256":h(lib),"metallib_sha256":h(metal)},indent=2)+"\n")
p=subprocess.Popen([str(probe),arm],cwd=R,env={**os.environ,"CROW_SUBSTEP_TRACE":"1"},stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def pump(src,path):
 with path.open("wb") as raw:
  with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as out:
   for chunk in iter(lambda:src.read(1<<16),b""):out.write(chunk)
ts=[threading.Thread(target=pump,args=(p.stdout,D/"trace.log.gz")),threading.Thread(target=pump,args=(p.stderr,D/"substeps.log.gz"))]
for t in ts:t.start()
rc=p.wait()
for t in ts:t.join()
(D/"run-result.json").write_text(json.dumps({"exit_code":rc},indent=2)+"\n");print(arm,"exit",rc,flush=True);raise SystemExit(rc)
