import pathlib,json,shlex,subprocess,hashlib,time
b=pathlib.Path("/tmp/numi-human-joint-owner-build-033")
c=next(x for x in json.loads((b/"compile_commands.json").read_text()) if x["file"].endswith("matter/src/numi_human.mm"))
a=shlex.split(c["command"]); old=b/a[a.index("-o")+1]; new=pathlib.Path("/tmp/numi-human-d37-current-abi15.o"); a[a.index("-o")+1]=str(new)
t=time.monotonic(); p=subprocess.run(a,cwd=c["directory"],capture_output=True,text=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r={"command":a,"exit_code":p.returncode,"seconds":time.monotonic()-t,"old_object_sha256":h(old),"new_object_sha256":h(new) if new.exists() else None,"diagnostics":p.stderr};r["exact_object_parity"]=r["old_object_sha256"]==r["new_object_sha256"]
pathlib.Path("/tmp/numi-human-d37-abi-check.json").write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
