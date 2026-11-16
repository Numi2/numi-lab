import hashlib,json,math,struct,subprocess,sys,tempfile
from pathlib import Path
p=Path(__file__).resolve().parent;blob=(p/'articulated-kkt-input.bin').read_bytes();oracle=json.loads((p/'articulated-kkt-oracle.json').read_text());assert hashlib.sha256(blob).hexdigest()==oracle['input_sha256'];assert len(blob)==4*22496
with tempfile.TemporaryDirectory() as td:
 output=Path(td)/'output.bin';cmd=[sys.argv[1],str(p/'articulated-kkt.metal'),str(p/'articulated-kkt-input.bin'),str(output)];r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,(r.stdout,r.stderr);data=output.read_bytes()
assert len(data)==4*308
mapped={}
for k,c in enumerate(oracle['cases']):
 raw=struct.unpack_from('<5624f',blob,k*22496);v=struct.unpack_from('<76fI',data,k*308);x=v[:73];lam=v[73:76];assert v[-1]==0 and all(math.isfinite(q) for q in v[:-1]);A=raw[:5329];b=raw[5329:5402];J=raw[5402:5621];target=raw[5621:5624]
 residual=max(abs(math.fsum(A[i*73+j]*x[j] for j in range(73))-b[i]-math.fsum(J[a*73+i]*lam[a] for a in range(3))) for i in range(73));equality=max(abs(math.fsum(J[a*73+j]*x[j] for j in range(73))-target[a]) for a in range(3));difference=max(abs(a-b) for a,b in zip(x,c['reference']));assert residual<.01 and equality<1e-9 and difference<1e-6,(c['name'],residual,equality,difference);mapped[c['name']]=[x[j] for j in c['inverse_permutation']]+list(x[27:]);print(c['name'],'force_or_torque_residual',residual,'attachment_error_m',equality,'difference',difference)
for name in ('rest','incoming'):
 gap=max(abs(a-b) for a,b in zip(mapped[name+'_forward'],mapped[name+'_reverse']));assert gap<1e-6,(name,gap)
print('experimental articulated-rod Metal reference passed; not MetalWorld')
