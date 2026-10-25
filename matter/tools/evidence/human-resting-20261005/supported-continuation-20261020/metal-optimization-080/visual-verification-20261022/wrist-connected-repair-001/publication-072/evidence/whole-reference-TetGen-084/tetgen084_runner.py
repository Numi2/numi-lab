import hashlib, json, pathlib, socket, subprocess, time
D=pathlib.Path("/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded")
T=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/arm-local-pocket-remesh-116/local-remesh-118/tetgen-v1.6.0/tetgen")
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert not (D/"tetgen-run.json").exists(), "refuse overwrite"
for name in ("whole-skin-fat-reference.poly", "facet-marker-map.csv", "region-seeds.json"):
    assert (D/name).is_file()
cmd=[str(T), "-pq1.2/10AS128", "whole-skin-fat-reference.poly"]
start=time.monotonic()
try:
    cp=subprocess.run(cmd,cwd=D,text=True,capture_output=True,timeout=45)
    rc=cp.returncode; out=cp.stdout; err=cp.stderr; timed=False
except subprocess.TimeoutExpired as e:
    rc=None; out=e.stdout.decode(errors="replace") if isinstance(e.stdout,bytes) else (e.stdout or "")
    err=e.stderr.decode(errors="replace") if isinstance(e.stderr,bytes) else (e.stderr or "")
    timed=True
report={"status":"EXECUTED_DIAGNOSTIC_ONLY","host":socket.gethostname(),"command":cmd,"cwd":str(D),"timeout_seconds":45,"timed_out":timed,"return_code":rc,"elapsed_seconds":time.monotonic()-start,"stdout":out,"stderr":err,"binary_sha256":sha(T),"input_sha256":{n:sha(D/n) for n in ("whole-skin-fat-reference.poly","facet-marker-map.csv","region-seeds.json")},"switch_notes":{"p":"PLC tetrahedralization","q1.2/10":"quality/volume constraints","A":"assign region attributes","S128":"TetGen Steiner budget; independently enforce actual output additions <=128","Y":"omitted to permit facet/edge recovery subdivisions"}}
(D/"tetgen-run.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in report.items() if k not in ("stdout","stderr")},indent=2,sort_keys=True))
