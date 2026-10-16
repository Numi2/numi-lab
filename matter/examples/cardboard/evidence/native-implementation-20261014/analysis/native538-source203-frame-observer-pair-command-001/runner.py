from pathlib import Path
import argparse,json,hashlib,subprocess,os,time,datetime,importlib.util,sys
Q=Path("/Users/n/cardboard-implementation-20261014-host-macmini");P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
args=argparse.ArgumentParser();args.add_argument("--execute",action="store_true");args=args.parse_args()
spec=json.loads((P/"command-spec.json").read_text());S=Path(spec["snapshot"]);D=Path(spec["build"])
assert sha(S/"frozen-source.json")==spec["frozen_sha256"]
assert sha(D/"source-build.json")==spec["build_sha256"]
f=json.loads((S/"frozen-source.json").read_text());b=json.loads((D/"source-build.json").read_text())
assert b["success"] and b["source_sha256"]==f["source_sha256"]
for n,h in f["source_sha256"].items():assert sha(S/n)==h,n
for n,h in b["artifacts_sha256"].items():assert sha(n)==h,n
m=json.loads((P/"manifest.json").read_text())
for n,h in m.items():assert sha(P/n)==h,n
mod=importlib.util.spec_from_file_location("capacity538",P/"capacity-helper.py");cap=importlib.util.module_from_spec(mod);mod.loader.exec_module(cap);cap.B=Q
workers=[r for r in subprocess.check_output(["ps","-axo","pid=,stat=,comm="],text=True).splitlines() if "numi-matter-" in r and not r.split()[1].startswith("Z")]
pre={"at_utc":utc(),"source_files":len(f["source_sha256"]),"artifacts":len(b["artifacts_sha256"]),"workers":workers,"available_bytes":cap.available(),"execute_requested":args.execute,"qualification":"observer-prefix regression only"}
print(json.dumps(pre),flush=True)
if not args.execute:sys.exit(0)
assert not workers,workers
assert cap.available()>=cap.MINIMUM
E=P/"execution";E.mkdir()
results=[]
for name in ("on","off"):
 case=Path(spec["cases"][name]);assert not case.exists()
 argv=[str(Q/"venvs/cardboard-reference-185/bin/python3"),"-B",str(S/"matter/tools/cardboard_source_bound_run.py"),"--source-snapshot",str(S),"--runtime-dir",str(D),"--build-provenance",str(D/"source-build.json"),"--case",str(case),"--timeout","600","--question","Does exact native FP32 material-frame capture preserve the production control0 history under the bounded repeated-fallback solver?","--prediction","Both fresh control0 prefixes retain identical physical states; the observer exports native frame matrix rows and independent compensated reconstruction matches captured paper F at center and four sides. No full coupon qualification.","--diagnostic-run"]
 env=spec["environment"].copy()
 if name=="on":
  argv+=["--diagnostic-capture","NM_FEM_OPERATOR_FD_CAPTURE_PATH"]
  env["NM_FEM_OPERATOR_FD_SELECTOR"]="0:0"
 for k,v in sorted(env.items()):argv+=["--env",k+"="+v]
 for inp in (P/"command-spec.json",S/"frozen-source.json"):argv+=["--input-file",str(inp)]
 native=[a.replace("{CASE}",str(case)) for a in spec["native_argv"]]
 argv+=["--"]+native
 run=E/name;run.mkdir();(run/"launch.json").write_text(json.dumps({"at_utc":utc(),"argv":argv},indent=2)+"\n")
 reserve=run/"emergency-reserve.bin";proc=None;errors=[];timed=False;space=False;start=time.monotonic()
 try:
  cap.write_reserve_file(reserve)
  clean={k:v for k,v in os.environ.items() if not k.startswith(("NM_","NUMI_MATTER_","NUMI_CARDBOARD_"))};clean.update(PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",PYTHONDONTWRITEBYTECODE="1")
  with (run/"wrapper.log").open("xb") as log:
   proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=clean,start_new_session=True)
   while proc.poll() is None:
    space=cap.available()<cap.ABORT_FLOOR;timed=time.monotonic()-start>720
    if space or timed:reserve.unlink(missing_ok=True);errors.extend(cap.terminate_owned_native(proc,native[0])["cleanup_errors"]);break
    time.sleep(2)
 except BaseException as ex:
  errors.append(repr(ex))
  if proc is not None and proc.poll() is None:errors.extend(cap.terminate_owned_native(proc,native[0])["cleanup_errors"])
 finally:reserve.unlink(missing_ok=True)
 unchanged=all(sha(S/n)==h for n,h in f["source_sha256"].items()) and all(sha(n)==h for n,h in b["artifacts_sha256"].items())
 result={"at_utc":utc(),"case":str(case),"exit":None if proc is None else proc.poll(),"wall_s":time.monotonic()-start,"errors":errors,"timed_out":timed,"space_abort":space,"source_artifacts_unchanged":unchanged,"full_coupon_qualified":False}
 (run/"receipt.json").write_text(json.dumps(result,indent=2)+"\n");results.append(result);print(json.dumps(result),flush=True)
 if not unchanged or space:break
(E/"receipt.json").write_text(json.dumps({"at_utc":utc(),"cases":results,"full_coupon_qualified":False},indent=2)+"\n")
sys.exit(0 if len(results)==2 and all(r["exit"]==0 and not r["errors"] and not r["timed_out"] and not r["space_abort"] and r["source_artifacts_unchanged"] for r in results) else 2)
