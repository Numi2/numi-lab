from pathlib import Path
import json,hashlib,subprocess,os,argparse,datetime,time,sys
Q=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
P=Path(__file__).resolve().parent;H=Q/"analysis/native531-armijo-cgls-fallback-helper-002"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((H/"package-manifest.json").read_text())
assert sha(H/"package-manifest.json")=="d61c655ce924303dcd0e66bc0eff24a42a621d93bb2c366442683bfd7caf973b"
for rec in m["files"]:assert sha(H/rec["path"])==rec["sha256"] and (H/rec["path"]).stat().st_size==rec["size"],rec
S=Path(m["source_snapshot"]);B=Q/"build-source203-fallback-001"
assert sha(S/"frozen-source.json")==m["source_snapshot_manifest_sha256"]
assert sha(B/"source-build.json")==m["source_build_manifest_sha256"]
f=json.loads((S/"frozen-source.json").read_text());b=json.loads((B/"source-build.json").read_text())
assert f["source_sha256"]==b["source_sha256"] and b["success"]
for n,h in f["source_sha256"].items():assert sha(S/n)==h,n
for n,h in b["artifacts_sha256"].items():assert sha(n)==h,n
for n,h in json.loads((P/"manifest.json").read_text()).items():assert sha(P/n)==h,n
a=argparse.ArgumentParser();a.add_argument("--execute",action="store_true");args=a.parse_args()
workers=[r for r in subprocess.check_output(["ps","-axo","pid=,stat=,comm="],text=True).splitlines() if "numi-matter-" in r and not r.split()[1].startswith("Z")]
print(json.dumps({"workers":workers,"source_files_verified":len(f["source_sha256"]),"helper_files_verified":len(m["files"]),"gpu_requested":args.execute}),flush=True)
if not args.execute:sys.exit(0)
assert not workers,workers
out=Q/"native/armijo-cgls-fallback-537";out.mkdir(exist_ok=False);results={}
commands=json.loads((H/"build/command.json").read_text())
for opt in ("O0","O3"):
 r=out/opt;r.mkdir();start=time.monotonic();errors=[];code=None;timed=False
 try:
  with (r/"stdout.json").open("xb") as so,(r/"stderr.log").open("xb") as se:
   p=subprocess.run(commands[opt],stdout=so,stderr=se,timeout=60,env={**os.environ,"PATH":"/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"})
   code=p.returncode
 except subprocess.TimeoutExpired:timed=True
 except BaseException as e:errors.append(repr(e))
 receipt={"at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"argv":commands[opt],"exit_code":code,"timed_out":timed,"errors":errors,"wall_s":time.monotonic()-start,"stdout_sha256":sha(r/"stdout.json"),"stderr_sha256":sha(r/"stderr.log")}
 (r/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n");results[opt]=receipt
unchanged=all(sha(S/n)==h for n,h in f["source_sha256"].items()) and all(sha(n)==h for n,h in b["artifacts_sha256"].items()) and all(sha(H/r["path"])==r["sha256"] for r in m["files"])
same=(out/"O0/stdout.json").read_bytes()==(out/"O3/stdout.json").read_bytes()
report={"at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"results":results,"stdout_bitwise_equal":same,"source_helper_build_unchanged":unchanged,"full_fem_transactions":0,"full_coupon_qualification":False,"scope":"Actual Armijo fallback eligibility, runtime layout/initializers and CGLS recurrence, synthetic A/Atranspose"}
(out/"run-receipt.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report),flush=True)
sys.exit(0 if unchanged and same and all(r["exit_code"]==0 and not r["timed_out"] and not r["errors"] for r in results.values()) else 2)
