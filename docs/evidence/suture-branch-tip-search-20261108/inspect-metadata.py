import io,json,os,subprocess,tarfile
ids=["e48f0c6872c7ad7f4b627a7feeb7452ea31324f3","b154df8dfe61bf393bba59b3d16da3197ae3a365"]
wanted={"inputs.sha256","outputs.sha256","launch.txt","source-revision.txt","source-status.txt","post-source-status.txt","precheck.log","postcheck.log","results/entry/partial-owner-capsule/manifest.tsv","results/advance/partial-owner-capsule/manifest.tsv"}
root="/Users/numi/numi-lab-policy-repair-main-20261108"; env=dict(os.environ,GIT_NO_LAZY_FETCH="1",GIT_OPTIONAL_LOCKS="0");out=[]
for oid in ids:
 z={"object":oid,"documents":[],"errors":[]}
 try:
  data=subprocess.run(["git","-C",root,"cat-file","blob",oid],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30,check=True).stdout
  with tarfile.open(fileobj=io.BytesIO(data),mode="r:gz") as t:
   for m in t:
    name=m.name.removeprefix("./")
    if name not in wanted:continue
    if not m.isfile() or m.size>65536:raise ValueError("invalid bounded document "+name)
    z["documents"].append({"name":name,"size":m.size,"text":t.extractfile(m).read().decode("utf-8","replace")})
 except Exception as e:z["errors"].append(repr(e))
 out.append(z)
print(json.dumps(out,indent=2))
