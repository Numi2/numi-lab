#!/usr/bin/env python3
import hashlib, json, pathlib, shlex, shutil, subprocess
W=pathlib.Path("/Users/n/numi-lab-native-replay-publish-20261017")
ROOT=pathlib.Path("/Users/n/numi-human-separation-20261018/C/outer-contact-mpm-261")
BUILD=ROOT/"build"; SRC=BUILD/"source"
BASE=pathlib.Path("/Users/n/numi-human-separation-20261018/C/current-operator-integration-207/full-source")
BASEBUILD=pathlib.Path("/Users/n/numi-human-separation-20261018/C/current-operator-integration-207/full-build")
CMDJSON=pathlib.Path("/Users/n/numi-human-separation-20261018/C/current-operator-integration-207/full-library-commands-base.json")
LINKJSON=pathlib.Path("/Users/n/numi-human-separation-20261018/C/aortic-wall-outer-contact-260/production-outer-contact-001/build-runtime-001/link-invocation.json")
PATCHSRC=ROOT/"source"
if BUILD.exists(): raise SystemExit("refusing existing build dir")
BUILD.mkdir(parents=True); shutil.copytree(BASE,SRC,dirs_exist_ok=True)
# Refresh source and public headers from exact HEAD, avoiding stale 207 header copies.
tracked=[]
for tree in ("include","src","matter/include","matter/src"):
    tracked.extend(subprocess.check_output(["git","-C",str(W),"ls-tree","-r","--name-only","HEAD",tree],text=True).splitlines())
for rel in sorted(set(tracked)):
    dest=SRC/rel; dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(subprocess.check_output(["git","-C",str(W),"show",f"HEAD:{rel}"]))
# Overlay only the four authored product changes before compiling any ABI consumer.
for rel in ("matter/include/numi/matter/matter.hpp","matter/src/compiler_impl.cpp","matter/src/validation_impl.cpp","matter/src/runtime.mm"):
    (SRC/rel).write_bytes((PATCHSRC/rel).read_bytes())
commands=json.loads(CMDJSON.read_text()); selected=[]
for x in commands:
    relout=pathlib.Path(x["output"]); dep=BASEBUILD/(str(relout)+".d")
    if dep.exists() and "/matter/include/numi/matter/matter.hpp" in dep.read_text(errors="replace"):
        selected.append(x)
if len(selected)!=21: raise SystemExit(f"expected 21 ABI-dependent TUs, found {len(selected)}")
oldroot="/Users/n/numi-lab-complete-body-capture-1276"
objdir=BUILD/"objects"; depdir=BUILD/"deps"; logdir=BUILD/"logs"
for d in (objdir,depdir,logdir): d.mkdir()
newobjects={}; compile_rows=[]
for i,x in enumerate(selected):
    rel=pathlib.Path(x["file"]).relative_to(oldroot); src=SRC/rel
    if not src.exists(): raise SystemExit(f"missing source: {src}")
    argv=[a.replace(oldroot,str(SRC)) for a in shlex.split(x["command"])]
    if src.name == "numi_human.mm":
        ap=pathlib.Path("/Users/n/numi-human-separation-20261018/C/aortic-wall-outer-contact-260/production-outer-contact-001/build-runtime-001/adapter-compile-invocation.json")
        adapter=json.loads(ap.read_text())["argv"]
        guards=[a for a in adapter if a.startswith("-DNUMI_HUMAN_TENDON_FEM_TEST_SOURCE")]
        if len(guards)!=2: raise SystemExit("missing exact 260 Human shader source guard macros")
        argv[1:1]=guards
    if src.name == "runtime.mm": argv.insert(1,"-DNM_MATTER_TEST_SOURCE_LOADER=1")
    argv[argv.index("-c")+1]=str(src)
    obj=objdir/f"{i:02d}-{src.stem}.o"; dep=depdir/f"{i:02d}-{src.stem}.d"
    oi=argv.index("-o"); argv[oi+1]=str(obj)
    if "-MF" in argv: argv[argv.index("-MF")+1]=str(dep)
    else: argv[oi:oi]=["-MMD","-MF",str(dep)]
    if not obj.is_relative_to(BUILD) or not dep.is_relative_to(BUILD): raise SystemExit("output escaped isolated build")
    p=subprocess.run(argv,cwd=BUILD,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    log=logdir/f"{i:02d}-{src.stem}.log"; log.write_text(p.stdout)
    if p.returncode: raise SystemExit(f"compile failed {rel}: {p.returncode}; log={log}")
    newobjects[x["output"]]=str(obj)
    compile_rows.append({"source":str(src),"source_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),"argv":argv,"object":str(obj),"object_sha256":hashlib.sha256(obj.read_bytes()).hexdigest(),"depfile":str(dep),"exit_code":p.returncode})
# Link against the tested 260 closure, replacing every object that used matter.hpp by basename.
link=json.loads(LINKJSON.read_text())["argv"]
out=BUILD/"libmetalrobo.dylib"; link[link.index("-o")+1]=str(out)
replaced=[]
for x in selected:
    wanted=pathlib.Path(x["output"]).name
    matches=[j for j,a in enumerate(link) if pathlib.Path(a).name==wanted]
    if len(matches)!=1: raise SystemExit(f"expected one link input named {wanted}, found {len(matches)}")
    j=matches[0]; old=link[j]; link[j]=newobjects[x["output"]]; replaced.append({"old":old,"new":link[j]})
if len(replaced)!=21: raise SystemExit("incomplete ABI object replacement")
p=subprocess.run(link,cwd=BUILD,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(logdir/"link.log").write_text(p.stdout)
if p.returncode: raise SystemExit(f"link failed {p.returncode}; see {logdir / link.log}")
header=str(SRC/"matter/include/numi/matter/matter.hpp")
for row in compile_rows:
    if header not in pathlib.Path(row["depfile"]).read_text(errors="replace"):
        raise SystemExit(f"candidate ABI header absent from depfile: {row[source]}")
manifest={"base_head":subprocess.check_output(["git","-C",str(W),"rev-parse","HEAD"],text=True).strip(),"candidate_header_sha256":hashlib.sha256(pathlib.Path(header).read_bytes()).hexdigest(),"source_objects":compile_rows,"replaced_link_objects":replaced,"link_argv":link,"library":str(out),"library_sha256":hashlib.sha256(out.read_bytes()).hexdigest()}
(BUILD/"build-receipt.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"compiled_abi_tus":len(compile_rows),"library":str(out),"sha256":manifest["library_sha256"]},indent=2))
