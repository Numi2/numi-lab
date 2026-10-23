from pathlib import Path
import re,json,gzip,hashlib
D=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/sparse-bone-contact-135")
def fields(l):return dict(re.findall(r"(\w+)=([^ ]+)",l))
r={}
for name in ["baseline-matched-001","sparse-matched-001"]:
 N=D/name;ls=(N/"stdout.log").read_text().splitlines();term=fields(next(l for l in ls if l.startswith("resting_FEM_terminal_audit=")));body=fields(next(l for l in ls if l.startswith("resting_integrated_body=")))
 r[name]={"terminal":term,"body":body,"execution":json.loads((N/"execution-receipt.json").read_text())}
 for filename in ["stdout.log","stderr.log"]:
  with gzip.GzipFile(str(N/(filename+".gz")),"wb",mtime=0) as f:f.write((N/filename).read_bytes())
keys=["min_J","max_J","max_displacement_m","max_anchor_reaction_N"]
r["relative_endpoint_changes"]={k:abs(float(r["sparse-matched-001"]["terminal"][k])-float(r["baseline-matched-001"]["terminal"][k]))/max(abs(float(r["baseline-matched-001"]["terminal"][k])),1e-30) for k in keys}
r["all_endpoints_within_one_percent"]=all(v<=.01 for v in r["relative_endpoint_changes"].values());r["both_eight_steps_accepted"]=all(r[n]["terminal"]["accepted_steps"]=="8" for n in ["baseline-matched-001","sparse-matched-001"])
r["scope"]="1ms matched native mechanical regression only; no skin/ROM or sustained coupled performance qualification"
(D/"native-comparison.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
