from pathlib import Path
import hashlib,json,struct,subprocess,sys
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-split-precision-20261108';B=R/'build/crow-split-precision-build-01'
sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects,content
assert not B.exists();B.mkdir()
def digest(x):return hashlib.sha256(x).hexdigest()
entries=tree()
with Objects() as objects:
    m=json.loads(objects.read(entries['artifacts/crow-rl-contact-precision-20261108/payload/manifest.json'][1]))
    reference=objects.read(entries['artifacts/crow-rl-contact-precision-20261108/reference-source.py'][1])
    (A/'independent-reference-source.py').write_bytes(reference)
    captures={}
    for name,suffix in [('substep7','/substep7-capture.bin'),('substep2','/control-substep2.bin')]:
        f=next(f for f in m['files'] if f['path'].endswith(suffix))
        raw=b''.join(content(f,entries,objects));assert len(raw)==f['bytes'] and digest(raw)==f['sha256']
        (B/(name+'-capture.bin')).write_bytes(raw);captures[name]=raw
def scalar(at,reg,lo,hi):
    return struct.pack('<8I20f',at,1,0,0,0,0,0,0,*([1.]*8),reg,*([1.]*7),lo,hi,0.,0.)+bytes(16)
def cone(at):
    return struct.pack('<8I20f',at,3,1,0,0,0,0,0,*([1.]*8),*([1.]*8),0.,0.,0.,0.)+bytes(16)
cases=[]
def save(name,nv,rows,blocks,mass,jac,bias,free,start,rounding=0):
    raw=struct.pack('<8I',0x53504c31,1,nv,rows,len(blocks)//128,4,rounding,0)+blocks
    arrays=[np.asarray(x,dtype='<f4').reshape(-1) for x in [mass,jac,bias,free,start]]
    assert [len(x) for x in arrays]==[nv*nv,rows*nv,rows,nv,nv]
    assert all(np.isfinite(x).all() for x in arrays)
    raw+=b''.join(x.tobytes() for x in arrays)
    path=B/(name+'.input');path.write_bytes(raw)
    cases.append(dict(name=name,path=str(path.relative_to(R)),bytes=len(raw),sha256=digest(raw),rounding=rounding,nv=nv,rows=rows,blocks=len(blocks)//128))
for name,raw in captures.items():
    h=struct.unpack_from('<8I',raw);assert h[0]==0x51464c31 and h[1]==1 and h[5]==len(raw)
    d=struct.unpack_from('<16I8f',raw,32);nv,rows,nb=d[2:5];offset=256+128*nb
    blocks=raw[256:offset]
    def take(n):
        global offset
        x=np.frombuffer(raw,dtype='<f4',count=n,offset=offset).copy();offset+=4*n;return x
    mass,jac,bias,free,warm,wp,failed=[take(n) for n in [nv*nv,rows*nv,rows,nv,nv,rows,nv]]
    for rounding in [0,1]:save(name+('-split' if not rounding else '-rounded'),nv,rows,blocks,mass,jac,bias,free,failed,rounding)
save('null',2,0,b'',np.eye(2),[],[],[.25,-.125],[.25,-.125])
save('spd',2,2,scalar(0,.25,-2,2)+scalar(1,.5,-2,2),[[2,.5],[.5,1]],np.eye(2),[-.25,.125],[.2,-.3],[0,0])
save('scalar-bound',1,1,scalar(0,.25,-.5,.5),[2],[1],[-4],[1],[0])
save('unsupported-boundary',3,3,cone(0),np.eye(3),np.eye(3),[0,1,0],[0,0,0],[0,0,0])
source_files=['split.metal','probe.mm','prepare.py','run.py','independent-reference-source.py']
plan=dict(schema='crow-native-split-precision-study-v1',
 question='Can native two-FP32 expansions retain a velocity solution at the existing 1e-10 gate when the same solve rounds each iterate to canonical FP32?',
 purpose='Exposed engineering representation experiment; no physical state advancement or ecological confirmation.',
 prior_evidence=dict(commit='0f1ed00dbc3d16f3fe0f8dbfc973f313dd432b44',local_proof='artifacts/crow-rl-contact-precision-20261108/local-box-proof.json',limits='Local FP32 infeasibility near rounded reference; failed native velocity lies outside certified box.'),
 mechanism='FP32 expansion row sums, projection, dynamics/constraint residual, Hessian, LDL and velocity update. No FP64 or reference state enters the Metal solve.',
 supported_instrument_scope='SPD A, scalar intervals, zero-adhesion uncapped cones in interior/vertex regions. Cone boundary/cap/adhesion returns explicit code3; no substituted projection.',
 predictions=dict(calibration_pass=True,split_captures_gate_pass=True,split_independent_scaled_residual_max=1e-10,split_velocity_reference_max_error=1e-10,split_reported_gradient_abs_error_max=2e-12,substep7_rounded_gate_pass=False,repeated_binary_outputs_exact=True,unsupported_boundary_code=3),
 calibration=dict(exact_fraction_arithmetic=['1+2^-30','subtraction','(1+2^-24)*(1-2^-24)','1/10','1e8+1-1e8','-7/3'],sqrt2_abs_error_max=2e-14,arithmetic_abs_error_max=2e-14,independent_weighted_cone_calibration='same previously calibrated FP64 reference; rerun five closed-form and weighted derivative checks'),
 experimental_units='Two already exposed fixed captured problems; synthetic controls. Repeat submissions test determinism, not independence.',
 intervention='Retain hi/lo state vs discard low part after each identical Newton update.',
 held_fixed=['compiled FP32 A/J/bias/free/start/block records','all four Newton updates','Metal device/compiler','same arithmetic except state rounding','existing 1e-10 stationarity target'],
 cases=cases,
 budget=dict(native_submissions=14,calibration_submissions=2,problem_submissions=12,seconds_each=30,raw_bytes_max=2000000,new_source_bytes_max=50000),
 commands=dict(build_metal=['xcrun','-sdk','macosx','metal','-std=metal3.1','-O3','-fno-fast-math','-Iinclude','-c',str(A.relative_to(R)/'split.metal'),'-o',str(B.relative_to(R)/'split.air')],
 build_library=['xcrun','-sdk','macosx','metallib',str(B.relative_to(R)/'split.air'),'-o',str(B.relative_to(R)/'split.metallib')],
 build_probe=['xcrun','clang++','-std=c++20','-O2','-fobjc-arc','-framework','Metal','-framework','Foundation',str(A.relative_to(R)/'probe.mm'),'-o',str(B.relative_to(R)/'probe')],
 run=[str(R/'python/.venv/bin/python'),str(A.relative_to(R)/'run.py')]),
 sources={n:dict(bytes=(A/n).stat().st_size,sha256=digest((A/n).read_bytes())) for n in source_files},
 stopping='No retries or extra candidates. First unexpected process/GPU/structural error stops submissions; preserve every failure. Scientific prediction failures retained and analyzed.',
 promotion=False,limits='A successful split solve alone cannot qualify canonical FP32 output or coupled world. Production needs an explicit end-to-end representation contract, physical validation and performance assessment; no tolerance relaxation.')
(A/'plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(cases=len(cases),input_bytes=sum(c['bytes'] for c in cases),sources_bytes=sum(x['bytes'] for x in plan['sources'].values()))))
