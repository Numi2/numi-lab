import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path("/Users/numi/numi-reconciliation-build-20261108")
assets=Path("/Users/numi/numi-v5-admission-artifacts-20261107/v5-continuation-aggregate-status-diagnostic")
expected={"rigid.nhrigid":"2c78cb4150b97cea6e8169dad9e8f5dd59af667b247e07b56e48e857435560e4","muscle.nhmyo":"e5bb8a8168706bb3b23cf42b1d849659569ba2e6027bcc85729c3110fdbb8c4b","support.nhcnt":"4d54f8155cd83baaee7af536099824ac0da61e5d5e77544b42c6e5ce1b48c907","equality.nheq":"c99f69b6c217e0bce518cb4db7b27416be0d36f9a1cef26490cf8773850847df","retained-visual.mrvpack":"b8ab7cd8deead1e2af3137a727eaad74cd7e4238d77d9116d774bbf304936e37","retained-vision.json":"166220533c6c78695f6e55412f2e832eaaa4ec94ddf791b476111dc323c652a6","synthetic.nmatterpack":"0187d2ce30cac21273b1cdbaee6c00ed373a4c428354d2295719e8fe39975a6a","synthetic.nhtbind":"0a8c24b4e1ae919188095858763698d9f2cfa5991a64954015030da201c2a59d","synthetic.nhcartilage":"4241e17f251f70b279d25dcf97b2484ed81f77bf6597f5ca39c7e9b97d3d8374"}
checks={k:{"path":str(assets/k),"expected":v,"actual":hashlib.sha256((assets/k).read_bytes()).hexdigest()} for k,v in expected.items()}
Path("v5-asset-hashes.json").write_text(json.dumps(checks,indent=2)+"\n")
if any(x["actual"]!=x["expected"] for x in checks.values()):raise SystemExit("fixture source hash mismatch; no native tests run")
shader=str(root/"matter/shaders/NumiMatter.metallib");adapter=str(root/"numi_human_costal_ownership_native");owner=str(root/"numi_human_costal_real_owner_probe");v5=str(root/"numi_human_costal_v5_admission_probe")
args=[str(assets/k) for k in expected]+[str(root/"shaders/MetalRobo.metallib")]
cases=[("adapter-empty",[adapter,shader,"--adapter-control"]),("adapter-contact",[adapter,shader,"--adapter-control","--body-contact"]),("owner-reject",[owner]),("owner-accept",[owner,"--accept"]),("owner-replay",[owner,"--replay"]),("owner-continue",[owner,"--continue"]),("owner-bounded",[owner,"--bounded"]),("owner-rollback",[owner,"--owner-rollback"]),("owner-rollback-repeat",[owner,"--owner-rollback"]),("v5-admission",[v5]+args),("v5-admission-repeat",[v5]+args)]
env=os.environ.copy();env["MRNX_MATTER_METALLIB"]=shader
out=[]
for name,cmd in cases:
 try:
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=120,env=env);record={"name":name,"command":cmd,"exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
 except subprocess.TimeoutExpired as e:
  record={"name":name,"command":cmd,"timeout":True,"stdout":(e.stdout or b"").decode(errors="replace") if isinstance(e.stdout,bytes) else (e.stdout or ""),"stderr":(e.stderr or b"").decode(errors="replace") if isinstance(e.stderr,bytes) else (e.stderr or "")}
 out.append(record);Path("native-final.json").write_text(json.dumps(out,indent=2)+"\n");print(name,record.get("exit","TIMEOUT"),flush=True)
