from pathlib import Path
import json,hashlib,subprocess,os,time,datetime,signal,sys
q=Path("/Users/n/cardboard-implementation-20261014-host-macmini");s=q/"snapshots/source-integrated-configure-support-219-001";b=q/"build-source219-integrated-001";c=q/"native/integrated-contact-baseline-549"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((s/"frozen-source.json").read_text());cfg=json.loads((s/"configure-support-source.json").read_text());bm=json.loads((b/"source-build.json").read_text())
assert sha(s/"frozen-source.json")=="8874cc2b52e6d2a0125b2cb91ee31566b1d81d0dd10481d1ac85a5cae30e39f3"
assert sha(b/"source-build.json")=="d248bac88f755345f1d7f7bd724b6a0a06bd3dfd4a9ef7a4c3414bf47bc1ae76" and bm["success"] and bm["source_sha256"]==f["source_sha256"]
def verify():
 return all(sha(s/n)==h for n,h in f["source_sha256"].items()) and all(sha(s/n)==h for n,h in cfg["files"].items()) and all(sha(n)==h for n,h in bm["artifacts_sha256"].items())
assert verify()
workers=[l for l in subprocess.check_output(["ps","-axo","pid=,stat=,comm="],text=True).splitlines() if any(x in l for x in ["numi-matter-","generic-batch","geometry-replay"]) and not l.split()[1].startswith("Z")]
assert not workers,workers
c.mkdir(exist_ok=False)
a=[str(b/"matter/numi-matter-cardboard-full-contact-armijo-check"),str(s/"matter/materials/nagasawa2013_medium_orthotropic.nmatter"),"--sparse-free-body-parity","--diagnostic-state-output",str(c/"accepted-state-output")]
env={"PATH":"/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin","HOME":"/Users/n","NUMI_MATTER_METALLIB":str(b/"matter/shaders/NumiMatter.metallib")}
(c/"command.json").write_text(json.dumps({"schema":"numi.cardboard.native549-command.v1","argv":a,"environment":env,"source_frozen_sha256":sha(s/"frozen-source.json"),"source_build_sha256":sha(b/"source-build.json"),"configure_support_manifest_sha256":sha(s/"configure-support-source.json"),"all_inputs_verified":verify(),"predecessor546_refused_standalone_before_physics":True,"timeout_s":900,"question":"Does the identical integrated contact fixture pass with true-residual override disabled, separating the failed548setup from its contact sequence?","prediction":"No pass assumed; retain all accepted and rejected histories; small fixture only."},indent=2)+"\n")
start=time.monotonic();timed=False;code=None
with (c/"stdout.log").open("xb") as out,(c/"stderr.log").open("xb") as err:
 p=subprocess.Popen(a,env=env,stdout=out,stderr=err,start_new_session=True)
 print(json.dumps({"pid":p.pid,"case":str(c)}),flush=True)
 try:code=p.wait(timeout=900)
 except subprocess.TimeoutExpired:
  timed=True;os.killpg(p.pid,signal.SIGTERM)
  try:code=p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=10)
r={"schema":"numi.cardboard.native549-receipt.v1","exit_code":code,"timed_out":timed,"wall_seconds":time.monotonic()-start,"all_inputs_unchanged":verify(),"stdout_sha256":sha(c/"stdout.log"),"stderr_sha256":sha(c/"stderr.log"),"at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"physical_transactions":"adjudicate from retained full logs and accepted-state bindings; do not infer from exit","shipping_qualified":False}
(c/"receipt.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r),flush=True);sys.exit(code or int(timed or not r["all_inputs_unchanged"]))
