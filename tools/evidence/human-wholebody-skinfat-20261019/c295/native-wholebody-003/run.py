import hashlib,json,os,pathlib,subprocess,sys,time
root=pathlib.Path(__file__).resolve().parent
inv=json.loads((root/"invocation.json").read_text())
launch=inv["launcher_argv"]
assert "--" in launch and launch[launch.index("--")+1:]==inv["argv"], "exclusive wrapper argv is stale"
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
for key,pathkey in [("binary_sha256","argv"),("library_sha256","library_path"),("msl_sha256","msl_path")]:
    path=inv[pathkey][0] if pathkey=="argv" else inv[pathkey]
    assert sha(path)==inv[key], (key,path)
out=pathlib.Path(inv["argv"][4]).resolve()
assert out.resolve()==(root/"output").resolve()
assert not any(out.iterdir()), "output directory is not empty"
t=time.monotonic()
try:
    with (root/"stdout.log").open("wb") as so,(root/"stderr.log").open("wb") as se:
        p=subprocess.run(inv["launcher_argv"],cwd=inv.get("cwd"),env={**os.environ,**inv["environment"]},stdout=so,stderr=se,timeout=240,check=False)
    code=p.returncode; timed_out=False
except subprocess.TimeoutExpired:
    code=124; timed_out=True
def hfile(p):
    p=pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() and p.is_file() else None
files={}
for p in sorted(out.rglob("*")):
    if p.is_file(): files[str(p.resolve().relative_to(root.resolve()))]={"sha256":hfile(p),"bytes":p.stat().st_size}
result={"exit_code":code,"timed_out":timed_out,"elapsed_s":time.monotonic()-t,"binary_sha256":sha(inv["argv"][0]),"library_sha256":sha(inv["library_path"]),"msl_sha256":sha(inv["msl_path"]),"stdout_sha256":hfile(root/"stdout.log"),"stderr_sha256":hfile(root/"stderr.log"),"skin_reference_dump":hfile(root/"registered-skin.csv"),"outputs":files}
(root/"result.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
