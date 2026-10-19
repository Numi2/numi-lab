import hashlib,json,pathlib,shutil,subprocess,time,sys
root=pathlib.Path(__file__).resolve().parent
name=sys.argv[1]
build=root/name
assert not build.exists()
(build/"source/apps").mkdir(parents=True)
for p in (root/"source/apps").iterdir():
 if p.is_file():shutil.copy2(p,build/"source/apps"/p.name)
base=pathlib.Path("/Users/n/numi-human-separation-20261018/C/resting-construction-pose-292")
j=json.loads((base/"build-003/build-receipt.json").read_text())
oldbuild=str(base/"build-003");oldsource=str(base/"source/apps")
c=[x.replace(oldbuild,str(build)).replace(oldsource,str(build/"source/apps")) for x in j["compile_argv"]]
l=[x.replace(oldbuild,str(build)) for x in j["link_argv"]]
old_msl='/Users/n/numi-human-separation-20261018/C/contact-legacy-plane-bounded-recovery-279/MatterSource.metal'
new_msl='/Users/n/numi-human-separation-20261018/C/C298-material-column-self-contact/source/MatterSource-c279-c283.metal'
old_libdir='/Users/n/numi-human-separation-20261018/C/contact-legacy-plane-bounded-recovery-279/build-guarded-root-001'
new_libdir='/Users/n/numi-human-separation-20261018/C/tissue-mass-symmetric-301/build-runtime-001'
old_msl_arg='-DNUMI_MATTER_METALLIB="'+old_msl+'"'
new_msl_arg='-DNUMI_MATTER_METALLIB="'+new_msl+'"'
assert c.count(old_msl_arg)==1
c[c.index(old_msl_arg)]=new_msl_arg
old_rpath='-Wl,-rpath,'+old_libdir
old_dylib=old_libdir+'/libmetalrobo.dylib'
assert l.count(old_rpath)==1 and l.count(old_dylib)==1
l[l.index(old_rpath)]='-Wl,-rpath,'+new_libdir
l[l.index(old_dylib)]=new_libdir+'/libmetalrobo.dylib'
assert old_msl not in ' '.join(c) and old_libdir not in ' '.join(l)
rec={"compile_argv":c,"link_argv":l,"inputs":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (build/"source/apps").iterdir()}}
(build/"invocation.json").write_text(json.dumps(rec,indent=2))
t=time.monotonic()
with (build/"compile.stdout").open("wb") as so,(build/"compile.stderr").open("wb") as se:p=subprocess.run(c,stdout=so,stderr=se)
rec["compile_exit"]=p.returncode
if p.returncode==0:
 with (build/"link.stdout").open("wb") as so,(build/"link.stderr").open("wb") as se:p=subprocess.run(l,stdout=so,stderr=se)
 rec["link_exit"]=p.returncode
rec["elapsed_s"]=time.monotonic()-t
if p.returncode==0:
 binary=pathlib.Path(l[l.index("-o")+1]);rec["binary_path"]=str(binary);rec["binary_sha256"]=hashlib.sha256(binary.read_bytes()).hexdigest()
(build/"build-receipt.json").write_text(json.dumps(rec,indent=2))
print(json.dumps({k:v for k,v in rec.items() if k not in ("compile_argv","link_argv")},indent=2))
sys.exit(p.returncode)
