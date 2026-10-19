from pathlib import Path
import hashlib,json,subprocess,time
out=Path(__file__).parent
record=json.loads((out/"build-input.json").read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
app=out/"source/apps/numilab_human_myosim_visual_probe.mm"
require(sha(app)==record["candidate_app_sha256"],"candidate app changed")
for item in record["source_dependency_hashes"]:
    p=Path(item["path"])
    require(p.is_file() and sha(p)==item["sha256"],"compile dependency changed: "+str(p))
for phase in ("compile","link"):
    argv=record[phase+"_argv"]
    old_build=record["base_build"]
    require(old_build not in "\n".join(argv),"command retains a path into the base build")
    if phase=="compile":
        require(argv[argv.index("-o")+1]==str(out/"numilab_human_myosim_visual_probe.mm.o"),"compile object output is not the fenced target")
        require(Path(argv[argv.index("-c")+1])==app,"compile source is not the candidate app")
        require(Path(argv[argv.index("-MF")+1])==out/"human-native.d","dependency output escaped candidate build")
        require(not (out/"numilab_human_myosim_visual_probe.mm.o").exists(),"refusing existing object")
    else:
        require(str(out/"numilab_human_myosim_visual_probe.mm.o") in argv,"link does not consume candidate object")
        require(str(out/"numi-human-native")==argv[argv.index("-o")+1],"link output escaped candidate build")
        require(not (out/"numi-human-native").exists(),"refusing existing binary")
        oldlib=Path("/Users/n/numi-human-separation-20261018/C/continuum-integrated-234/build-cpu-006/libmetalrobo-233.dylib")
        require(oldlib.is_file() and sha(oldlib)=="71372f858367c5945eed34e7334f184f210f06c7cd21e6ec48cb340a4aebd60a","linked base library changed")
    started=time.monotonic()
    log=out/(phase+".log")
    with log.open("wb") as f: proc=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
    product=out/("numilab_human_myosim_visual_probe.mm.o" if phase=="compile" else "numi-human-native")
    result={"phase":phase,"argv":argv,"exit_code":proc.returncode,"elapsed_seconds":time.monotonic()-started,"log_path":str(log),"log_sha256":sha(log),"candidate_app_sha256":sha(app),"product_path":str(product),"product_sha256":sha(product) if product.is_file() else None}
    (out/(phase+"-receipt.json")).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
    if proc.returncode: raise SystemExit(proc.returncode)
