from pathlib import Path
import ast,hashlib,json,subprocess
root=Path(__file__).resolve().parent;repo=root.parents[2]
work=Path('/Users/numi/Work/mixed-joint-long-rod-artifacts-20261108'); out=root/'operator'; out.mkdir(exist_ok=True)
tmp=work/'operator';tmp.mkdir(exist_ok=True)
def literal(path,name):
 tree=ast.parse(path.read_text())
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return ast.literal_eval(n.value)
 raise ValueError(name)
inst=literal(root.parent/'mixed-full-operator-20261108/prepare.py','instrument').replace('dispatch.rodNodeCount == 3u && dispatch.rodEdgeCount == 2u','dispatch.rodNodeCount == 9u && dispatch.rodEdgeCount == 8u')
assert inst.count('value += dot(targetDirection,exact) - rodCrossContactResponse(')==1
factor=literal(root.parent/'mixed-full-operator-20261108/prepare_factor.py','factor').replace('dispatch.rodNodeCount == 3u && dispatch.rodEdgeCount == 2u','dispatch.rodNodeCount == 9u && dispatch.rodEdgeCount == 8u').replace('cache.velocityCount == 3u && cache.blockWidth == 2u','cache.velocityCount == 9u && cache.blockWidth == 8u').replace('idx < 85u','idx < 259u')
base=work/'long.metal'; s=base.read_text();start=s.index('    float laneNormalResidual = 0.0f;',s.index('const float groupError = simd_max(max(finalNormal, finalTangent));'));pos=s.index('        const float limitU =\n            cone.effectiveFrictionU * contact.impulses.x;',start)
common=literal(root.parent/'mixed-failing-row-witness-20261108/prepare.py','common'); witness=common+literal(root.parent/'mixed-failing-row-witness-20261108/prepare.py','a')+'            }\n        }\n'
variants={}
for axis in range(3):
 variants['axis'+str(axis)]=inst.replace('AXIS_VALUE',str(axis)+'u')
 variants['rigid'+str(axis)]=inst.replace('value += dot(targetDirection,exact) - rodCrossContactResponse(','value -= rodCrossContactResponse(').replace('AXIS_VALUE',str(axis)+'u')
for part in range(8):variants['factor'+str(part)]=factor.replace('PART_VALUE',str(part)+'u')
variants['witness']=witness
link=json.loads((root.parent/'mixed-failing-row-witness-20261108/build.json').read_text())[2]['command'];records=[];paths=[base,work/'rod9']
def run(label,cmd):
 print(label,flush=True); p=subprocess.run([str(x) for x in cmd],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 (out/(label+'.stdout')).write_text(p.stdout);(out/(label+'.stderr')).write_text(p.stderr)
 records.append({'label':label,'command':[str(x) for x in cmd],'returncode':p.returncode});(out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
 if p.returncode:raise SystemExit(p.returncode)
for name,insert in variants.items():
 shader=tmp/(name+'.metal');air=tmp/(name+'.air');lib=tmp/(name+'.metallib'); shader.write_text(s[:pos]+insert+s[pos:])
 run(name+'-compile',['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-I',repo/'include','-c',shader,'-o',air])
 run(name+'-link',[str(air) if x.endswith('/baseline.air') else str(lib) if x.endswith('/baseline.metallib') else x for x in link])
 run(name,[work/'rod9',lib,'2']);paths.extend([shader,air,lib])
(out/'artifacts.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p)+'\n' for p in paths))
