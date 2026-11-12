from pathlib import Path
import gzip,hashlib,json,os,subprocess,threading
R=Path.cwd();A=R/"artifacts/crow-v15-ankle-pitch-response-screen-20261112";B=R/"build/crow-v15-ankle-pitch-response-screen-20261112";P=R/"build/crow-v15-toe-velocity-substep-observer-20261111";W=R/"build/crow-v15-wide-contact-isolation-20261111"
cmd=["xcrun","clang++","-std=c++20","-O2","-I"+str(W/"include"),"-Iinclude",str(A/"probe.cpp"),"-L"+str(P),"-lmetalrobo","-Wl,-rpath,"+str(P),"-o",str(B/"probe")];subprocess.run(cmd,check=True);h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/"inputs.json").write_text(json.dumps({"compile":cmd,"probe_sha256":h(B/"probe"),"host_sha256":h(P/"libmetalrobo.dylib"),"metallib_sha256":h(W/"MetalRobo.metallib")},indent=2)+"\n")
for name,gain in [("positive","0.08"),("negative","-0.08")]:
 d=A/name;d.mkdir(exist_ok=False);p=subprocess.Popen([str(B/"probe"),gain],cwd=R,env={**os.environ,"CROW_SUBSTEP_TRACE":"1"},stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 def pump(src,path):
  with path.open("wb") as raw:
   with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as out:
    for b in iter(lambda:src.read(1<<16),b""):out.write(b)
 ts=[threading.Thread(target=pump,args=(p.stdout,d/"trace.log.gz")),threading.Thread(target=pump,args=(p.stderr,d/"substeps.log.gz"))]
 for t in ts:t.start()
 rc=p.wait()
 for t in ts:t.join()
 (d/"run-result.json").write_text(json.dumps({"gain":gain,"exit_code":rc},indent=2)+"\n");print(name,"exit",rc,flush=True)
