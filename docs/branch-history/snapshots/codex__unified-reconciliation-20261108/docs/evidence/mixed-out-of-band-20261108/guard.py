from pathlib import Path
import hashlib,json,subprocess
p=Path(__file__).resolve().parent;repo=p.parents[2];prev=repo/'docs/evidence/mixed-joint-long-rod-20261108'
work=Path('/Users/numi/Work/mixed-out-of-band-artifacts-20261108'); old=Path('/Users/numi/Work/mixed-joint-long-rod-artifacts-20261108/witness.metal')
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
assert any(line.split()[0]==sha(old) and line.split(maxsplit=1)[1]==str(old) for line in (prev/'witness/artifacts.sha256').read_text().splitlines())
s=old.read_text()
before='''            if (valid) valid=mrResearchRodWitnessesFitSelectedBand(
                witnessNodeA[0],witnessNodeB[0],witnessEdge[0],
                witnessNodeA[1],witnessNodeB[1],witnessEdge[1]);'''
after='''            if (valid) {
                const bool bandAllowed=mrResearchRodWitnessesFitSelectedBand(
                    witnessNodeA[0],witnessNodeB[0],witnessEdge[0],
                    witnessNodeA[1],witnessNodeB[1],witnessEdge[1]);
                for (uint i=0u;i<3u;++i)
                    contacts[constraintBase+ids[i]].friction.w=bandAllowed ? 7.0f : 8.0f;
                valid=bandAllowed;
            }'''
assert s.count(before)==1;s=s.replace(before,after)
shader=work/'guard.metal';air=work/'guard.air';library=work/'guard.metallib';shader.write_text(s)
link=json.loads((repo/'docs/evidence/mixed-failing-row-witness-20261108/build.json').read_text())[2]['command']
airs=[Path(x) for x in link if x.endswith('.air') and not x.endswith('/baseline.air')]
expected=(repo/'docs/evidence/mixed-full-operator-20261108/air-inputs.sha256').read_text()
old_hashes={line.split(maxsplit=1)[1]:line.split()[0] for line in expected.splitlines()}
current={str(f):sha(f) for f in airs}
changed={k:(old_hashes.get(k),v) for k,v in current.items() if old_hashes.get(k)!=v}
assert set(current)==set(old_hashes)
assert changed=={'/Users/numi/numi-reconciliation-build-20261108/shaders/MetalRoboHybridRenderer.air':('541b0fd9d1d922589d2900f1b4858b0f07fb7e422badf62975eefecf3c01cb6c','ea9b8641e6b13b8d5ab8ca9bd571816c144fa36a283137b8235147b3a608798a')}
(p/'current-air-inputs.sha256').write_text(''.join(sha(f)+'  '+str(f)+'\n' for f in airs))
controlair=work/'guard-control.air';controllib=work/'guard-control.metallib'
control_commands=[['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-I',str(repo/'include'),'-c',str(old),'-o',str(controlair)],
 [str(controlair) if x.endswith('/baseline.air') else str(controllib) if x.endswith('/baseline.metallib') else x for x in link],
 [str(work/'probe'),str(controllib),'2']]
for i,cmd in enumerate(control_commands):
 r=subprocess.run(cmd,cwd=repo,capture_output=True,text=True)
 (p/('guard-control-%d.stdout'%i)).write_text(r.stdout);(p/('guard-control-%d.stderr'%i)).write_text(r.stderr)
 if r.returncode:raise SystemExit(r.returncode)
assert json.loads((p/'guard-control-2.stdout').read_text())==json.loads((p/'witness.stdout').read_text()),'unmodified witness link control failed'
(p/'guard-control-commands.json').write_text(json.dumps(control_commands,indent=2)+'\n')
(p/'guard-control-artifacts.sha256').write_text(''.join(sha(f)+'  '+str(f)+'\n' for f in (controlair,controllib)))

commands=[['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-I',str(repo/'include'),'-c',str(shader),'-o',str(air)],
 [str(air) if x.endswith('/baseline.air') else str(library) if x.endswith('/baseline.metallib') else x for x in link],
 [str(work/'probe'),str(library),'2']]
records=[]
for i,cmd in enumerate(commands):
 r=subprocess.run(cmd,cwd=repo,capture_output=True,text=True)
 (p/('guard-%d.stdout'%i)).write_text(r.stdout);(p/('guard-%d.stderr'%i)).write_text(r.stderr)
 records.append({'command':cmd,'returncode':r.returncode});(p/'guard-commands.json').write_text(json.dumps(records,indent=2)+'\n')
 print(i,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
(p/'guard-inputs.sha256').write_text(sha(old)+'  '+str(old)+'\n'+expected)
(p/'guard-artifacts.sha256').write_text(''.join(sha(f)+'  '+str(f)+'\n' for f in (shader,air,library)))
