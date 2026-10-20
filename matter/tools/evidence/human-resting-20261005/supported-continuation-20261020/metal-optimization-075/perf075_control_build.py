import pathlib,json,subprocess,shlex
r=pathlib.Path("/Users/numi/numi-human-aortic-contact-opt-20261019");b=r/"build-integrated-validation-20261019/profile-build-039";c=r/"build-integrated-validation-20261019/perf075-control"
src=r/"matter/src/perf075_control_runtime.mm";src.write_bytes(subprocess.check_output(["git","show","ea2cbd84:matter/src/runtime.mm"],cwd=r))
cmd=next(x for x in json.load(open(b/"compile_commands.json")) if x["file"]==str(r/"matter/src/runtime.mm"))
a=shlex.split(cmd["command"]);a[a.index("-o")+1]=str(c/"runtime.mm.o");a[a.index(str(r/"matter/src/runtime.mm"))]=str(src)
subprocess.run(a,cwd=cmd["directory"],check=True)
link=subprocess.check_output(["ninja","-t","commands","lib/libmetalrobo.dylib"],cwd=b,text=True).splitlines()[-1];link=link.removeprefix(": && ")
# All other owner sources/objects are unchanged at the baseline revision.
a=shlex.split(link.split(" &&")[0]);a[a.index("-o")+1]=str(c/"libmetalrobo.dylib");a=[str(c/"runtime.mm.o") if x=="matter/CMakeFiles/numi_matter_runtime_objects.dir/src/runtime.mm.o" else x for x in a]
subprocess.run(a,cwd=b,check=True);src.unlink();print("source-bound control linked")
