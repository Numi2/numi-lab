from pathlib import Path
import hashlib
import importlib.util
import json

B=Path('/Users/home/cardboard-implementation-20261014')
O=B/'analysis/source149-right-scale-disabled462-parity-001'
O.mkdir(exist_ok=True)
L=B/'solver/post-release-default-zero-parity-459'
R=B/'solver/cgls-block-right-disabled-parity-462'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
T=B/'analysis/native450-cgls-candidate-audit-tool-001/audit_cgls_snapshot.py'
spec=importlib.util.spec_from_file_location('binding462',T)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
lp=json.loads((L/'prediction.json').read_text());rp=json.loads((R/'prediction.json').read_text())
ls=Path(lp['source_snapshot']);rs=Path(rp['source_snapshot'])
for run,snap in [(L,ls),(R,rs)]:
    module,_=audit.load_audit_module(snap)
    _,receipt,_,_=audit.bind_run(run,snap,module)
    assert receipt['exit_code']==2 and not receipt['timed_out'] and receipt['source_inputs_unchanged']
paths=[(L,R),(ls,rs),(Path(lp['runtime_dir']),Path(rp['runtime_dir']))]
left_state=json.loads((L/'base-cgls.bin.state.json').read_text())
right_state=json.loads((R/'base-cgls.bin.state.json').read_text())
oldfp=left_state['device_program_fingerprint_u64_hex']
newfp=right_state['device_program_fingerprint_u64_hex']
assert oldfp!=newfp
assert left_state['world_fingerprint_u64_hex']==right_state['world_fingerprint_u64_hex']
normalizations=[]
def norm_string(value):
    for left,right in paths:
        value=value.replace(str(left),'<'+left.name+'>').replace(str(right),'<'+left.name+'>')
    return value
def compare_json(a,b,context):
    def visit(x,y,path):
        if isinstance(x,dict) and isinstance(y,dict):
            y=dict(y)
            for key in y.keys()-x.keys():
                assert key in ('right_scale','cgls_right_scale') and y[key] is None,(context,path,key,y[key])
                normalizations.append({'context':context,'path':path+'/'+key,'kind':'new_null_disabled_right_scale'})
                del y[key]
            assert x.keys()==y.keys(),(context,path,x.keys(),y.keys())
            for key in x:visit(x[key],y[key],path+'/'+key)
        elif isinstance(x,list) and isinstance(y,list):
            assert len(x)==len(y),(context,path)
            for i,(xx,yy) in enumerate(zip(x,y)):visit(xx,yy,path+'/'+str(i))
        elif isinstance(x,str) and isinstance(y,str):
            if path in ('/device_program_fingerprint_u64_hex','/device_program_fingerprint'):
                assert (x,y)==(oldfp,newfp),(context,path,x,y)
                normalizations.append({'context':context,'path':path,'kind':'independently_bound_device_program_identity'})
            else: assert norm_string(x)==norm_string(y),(context,path,x,y)
        else: assert type(x)==type(y) and x==y,(context,path,x,y)
    visit(a,b,'')

left_files={str(p.relative_to(L/'output')):p for p in (L/'output').rglob('*') if p.is_file()}
right_files={str(p.relative_to(R/'output')):p for p in (R/'output').rglob('*') if p.is_file()}
assert left_files.keys()==right_files.keys()
rows=[]
for key,p in sorted(left_files.items()):
    q=right_files[key];same=p.read_bytes()==q.read_bytes()
    if not same:
        assert Path(key).name=='manifest.json',key
        compare_json(json.loads(p.read_text()),json.loads(q.read_text()),'output/'+key)
    rows.append({'path':key,'raw_equal':same,'left_sha256':sha(p),'right_sha256':sha(q)})
payloads={}
for key in ['base-cgls.bin','base-cgls.bin.state.bin','first-status5.bin','stdout.log']:
    assert (L/key).read_bytes()==(R/key).read_bytes(),key
    payloads[key]={'bytes':(L/key).stat().st_size,'sha256':sha(L/key),'raw_equal':True}
for key in ['base-cgls.bin.state.json','first-status5.json']:
    compare_json(json.loads((L/key).read_text()),json.loads((R/key).read_text()),key)
left_lines=[line for line in (L/'stderr.log').read_text().splitlines() if not line.startswith('cardboard_step_timing=')]
right_lines=[line for line in (R/'stderr.log').read_text().splitlines() if not line.startswith('cardboard_step_timing=')]
assert len(left_lines)==len(right_lines)
for index,(left,right) in enumerate(zip(left_lines,right_lines)):
    if norm_string(left)==norm_string(right):continue
    lk,lv=left.split('=',1);rk,rv=right.split('=',1)
    assert lk==rk
    compare_json(json.loads(lv),json.loads(rv),'stderr/'+str(index)+'/'+lk)
lf=json.loads((ls/'frozen-source.json').read_text())['source_sha256']
rf=json.loads((rs/'frozen-source.json').read_text())['source_sha256']
assert lf.keys()==rf.keys()
changes=[key for key in lf if lf[key]!=rf[key]]
assert set(changes)=={'matter/src/runtime.mm','matter/src/metal/base_cgls.metalinc','matter/src/metal/fgmres.metalinc'}
for snap,entries in [(ls,lf),(rs,rf)]:
    for key,h in entries.items():assert sha(snap/key)==h,key
report={'schema':'numi.cardboard.source149-disabled-full-attempt-parity.v1',
        'parity_pass':True,'source_changes':changes,'source_files_verified_per_snapshot':len(lf),
        'output_files':rows,'payloads':payloads,'normalizations':normalizations,
        'execution_identity':{'left':oldfp,'right':newfp,'treated_as_equal':False},
        'non_timing_stderr_lines':len(left_lines),
        'production_trace_counts':{key:sum(x.startswith(key+'=') for x in left_lines) for key in ['fem_residual_armijo_iteration','fem_true_residual_cycle','fem_base_cgls_direction']},
        'limits':['Both complete attempts fail and roll back control1 after one accepted control.',
                  'The block3 right scale is absent; this establishes disabled parity only.',
                  'No complete coupon, numerical convergence, material calibration, assembly, crush or impact qualification.']}
(O/'compare.py').write_bytes(Path(__file__).read_bytes())
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(O/'manifest.json').write_text(json.dumps({'inputs':{str(p):sha(p) for p in [T,L/'prediction.json',L/'receipt.json',R/'prediction.json',R/'receipt.json',L/'stderr.log',R/'stderr.log']},'files':{p.name:sha(p) for p in O.iterdir() if p.is_file()}},indent=2)+'\n')
print(json.dumps({'parity_pass':True,'output_count':len(rows),'trace_counts':report['production_trace_counts'],'normalization_count':len(normalizations)}))
