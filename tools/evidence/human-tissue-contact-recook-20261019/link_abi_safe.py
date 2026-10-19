#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess
ROOT=pathlib.Path("/Users/n/numi-human-separation-20261018/C/outer-contact-mpm-261")
BUILD=ROOT/"build"; SRC=BUILD/"source"
W=pathlib.Path("/Users/n/numi-lab-native-replay-publish-20261017")
BASEBUILD=pathlib.Path("/Users/n/numi-human-separation-20261018/C/current-operator-integration-207/full-build")
CMDJSON=pathlib.Path("/Users/n/numi-human-separation-20261018/C/current-operator-integration-207/full-library-commands-base.json")
LINKJSON=pathlib.Path("/Users/n/numi-human-separation-20261018/C/aortic-wall-outer-contact-260/production-outer-contact-001/build-runtime-001/link-invocation.json")
commands=json.loads(CMDJSON.read_text()); selected=[]
for x in commands:
    dep=BASEBUILD/(x["output"]+".d")
    if dep.exists() and "/matter/include/numi/matter/matter.hpp" in dep.read_text(errors="replace"):
        selected.append(x)
if len(selected)!=21: raise SystemExit(f"expected21 selected TUs, found {len(selected)}")
link=json.loads(LINKJSON.read_text())["argv"]
out=BUILD/"libmetalrobo.dylib"; link[link.index("-o")+1]=str(out)
replacements=[]; rows=[]
for i,x in enumerate(selected):
    source=pathlib.Path(x["file"]); obj=BUILD/"objects"/f"{i:02d}-{source.stem}.o"
    dep=BUILD/"deps"/f"{i:02d}-{source.stem}.d"
    if not obj.exists() or not dep.exists(): raise SystemExit(f"missing compiled ABI output {obj}")
    names={pathlib.Path(x["output"]).name,source.name+".o",source.stem+".o"}
    matches=[j for j,a in enumerate(link) if pathlib.Path(a).name in names and a.endswith(".o")]
    if len(matches)!=1: raise SystemExit(f"expected one logical link input for {source.name}; found {[link[j] for j in matches]}")
    j=matches[0]; old=link[j]; link[j]=str(obj)
    replacements.append({"old":old,"new":str(obj)})
    deptext=dep.read_text(errors="replace")
    header=str(SRC/"matter/include/numi/matter/matter.hpp")
    if header not in deptext: raise SystemExit(f"candidate header missing from {dep}: {header}")
    rows.append({"source":str(SRC/source.relative_to("/Users/n/numi-lab-complete-body-capture-1276")),"object":str(obj),"object_sha256":hashlib.sha256(obj.read_bytes()).hexdigest(),"depfile":str(dep)})
if len(replacements)!=21: raise SystemExit("incomplete link replacement")
log=BUILD/"logs"/"link-corrected.log"
p=subprocess.run(link,cwd=BUILD,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT); log.write_text(p.stdout)
if p.returncode: raise SystemExit(f"link failed {p.returncode}; see {log}")
receipt={"base_head":subprocess.check_output(["git","-C",str(W),"rev-parse","HEAD"],text=True).strip(),"abi_header_sha256":hashlib.sha256((SRC/"matter/include/numi/matter/matter.hpp").read_bytes()).hexdigest(),"compiled_abi_tus":rows,"replaced_link_objects":replacements,"link_argv":link,"library":str(out),"library_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"link_exit_code":p.returncode}
(BUILD/"build-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({"compiled_abi_tus":len(rows),"library":str(out),"sha256":receipt["library_sha256"]},indent=2))
