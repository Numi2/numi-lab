"""Freeze an unused engineering grid and exact retained failing inputs."""
from pathlib import Path
import importlib.util,json,shutil,subprocess,struct
import numpy as np
from oracle import rotate,local_segment,cast,calibrate,HALF
R=Path.cwd();B=R/'build/crow-ccd-translation-build-02';A=R/'artifacts/crow-rl-ccd-translation-no-floor-20261108'
s=importlib.util.spec_from_file_location('archive',R/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
assert not a.git('status','--porcelain')
A.mkdir(exist_ok=True)
assert not list(A.iterdir()),'Preparation destination must be empty'
entries=a.tree()
def restore(manifest,source,target):
    with a.Objects() as o:m=json.loads(o.read(entries[manifest][1]))
    f=next(x for x in m['files'] if x['path']==source);p=B/target;p.parent.mkdir(parents=True,exist_ok=True)
    with a.Objects() as o,p.open('xb') as out:
        for block in a.content(f,entries,o):out.write(block)
    p.chmod(f['mode']);assert a.file_hash(p)==f['sha256']
for n in ('clamp','open','zero-friction','refinement'):
    for suffix in ('.ccdsweep','-replay.bin','-replay.json'):
        restore('artifacts/crow-rl-beak-hold-capture-20261108/payload/manifest.json',f'build/crow-beak-hold-build-01/capture-{n}{suffix}',f'input/{n}{suffix}')
restore('artifacts/crow-rl-ccd-stationary-20261108/manifest.json','build/crow-ccd-stationary-build-01/cases_probe.mm','cases_probe.mm')
restore('artifacts/crow-rl-ccd-stationary-20261108/manifest.json','.numi/retained-runtime/ccd-stationary-20261108/bin/cases','cases_probe')
query_source=(B/'cases_probe.mm').read_text().replace('argc==4','argc==5').replace('CASES NEW_OUTPUT','CASES ALPHAS NEW_OUTPUT').replace('argv[3]','argv[4]')
query_source=query_source.replace(' MRMetalWorldContactDispatchGPU d{};', ' std::vector<mr_float4> alphas(count);std::ifstream alphaFile(argv[3],std::ios::binary);alphaFile.read(reinterpret_cast<char*>(alphas.data()),alphas.size()*sizeof(alphas[0]));require(bool(alphaFile)&&alphaFile.peek()==EOF,"Invalid query alpha bytes");\n MRMetalWorldContactDispatchGPU d{};')
query_source=query_source.replace('mr_world_resolve_ccd','mr_ccd_grid_query_probe')
query_source=query_source.replace(' auto queue=[device newCommandQueue];',' auto alphaBuffer=buffer(alphas.data(),alphas.size()*sizeof(alphas[0]));\n auto queue=[device newCommandQueue];')
query_source=query_source.replace('[encoder dispatchThreads:MTLSizeMake(count,1,1)', '[encoder setBuffer:alphaBuffer offset:0 atIndex:13];[encoder dispatchThreads:MTLSizeMake(count*4,1,1)')
query_source=query_source.replace('records.size()*sizeof(records[0]));require(bool(out)', 'std::size_t(count)*4*sizeof(mr_float4));require(bool(out)')
query_source=query_source.replace('pair_orders','query_positions').replace('\\":2}', '\\":4}')
(B/'query_probe.mm').write_text(query_source)
for src,dst in [('build/shaders/MetalRobo.metallib','baseline.metallib'),('build/shaders/CCDSweepProbe.metallib','baseline-query.metallib'),('build/bin/metalrobo_ccd_sweep_probe','sweep_probe')]:
    shutil.copy2(src,B/dst)
shader=Path('apps/ccd_sweep_probe.metal').read_text().replace('../src/metal/Collision.metal','Collision.candidate.metal')+'\n'+(B/'query_kernel.metal').read_text()
(B/'candidate_probe.metal').write_text(shader)
rng=np.random.default_rng(2651521901);current=[];future=[];groups=[];pred=[]
for group,count in [('stationary',512),('overlap',256),('rotating',256),('away',512),('grazing',1024),('crossing',512),('common-translation',512),('sphere-spin',512)]:
    for j in range(count):
        q=rng.normal(size=4);q=(q/np.linalg.norm(q)).astype(np.float32)
        center=rng.uniform(-.3,.3,3).astype(np.float32)
        axes=rng.choice(3,size=1+j%3,replace=False);n=np.zeros(3);n[axes]=rng.uniform(.25,1,len(axes))*rng.choice([-1,1],len(axes));n/=np.linalg.norm(n)
        support=rng.uniform(-.8,.8,3)*HALF;support[axes]=np.sign(n[axes])*HALF[axes]
        gap=rng.uniform(11e-6,200e-6) if group!='overlap' else -rng.uniform(.0001,.001)
        p=support+n*(float(np.float32(.008))+gap)
        tangent=rng.normal(size=3);tangent-=n*(tangent@n);tangent/=np.linalg.norm(tangent)
        delta=np.zeros(3)
        if group=='away':delta=n*rng.uniform(0,.01)+tangent*rng.uniform(.001,.1)
        if group=='grazing':delta=-n*rng.uniform(1e-5,.005)+tangent*rng.uniform(.001,.15)
        if group=='crossing':delta=-2*p
        if group=='common-translation':delta=-n*rng.uniform(0,.002)+tangent*rng.uniform(0,.01)
        if group=='sphere-spin':delta=n*rng.uniform(-.0001,.0001)+tangent*rng.uniform(0,.002)
        c=np.zeros((2,5,4),dtype='<f4');c[:,2,3]=1;c[0,2]=q;c[0,1,:3]=center;c[0,1,3]=.0463680923
        c[1,1,:3]=center+rotate(q.astype(float),p);c[1,1,3]=.008
        f=c.copy();f[1,1,:3]=c[1,1,:3]+rotate(q.astype(float),delta)
        if group=='common-translation':
            common=rng.uniform(-.05,.05,3);f[:,1,:3]+=common
        if group=='sphere-spin':
            sphere=rng.normal(size=4);f[1,2]=(sphere/np.linalg.norm(sphere)).astype(np.float32)
            c[1,0,2]=np.asarray([rng.uniform(.001,.01)],dtype='<f4').view('<u4')[0].view('<f4')
        if group=='rotating':
            if j%2:f[0,2,0]=np.nextafter(f[0,2,0],np.float32(np.inf))
            else:
                v=rng.normal(size=4);f[0,2]=(v/np.linalg.norm(v)).astype(np.float32)
        current.append(c);future.append(f);groups.append(group)
        pred.append(None if group=='rotating' else cast(*local_segment(c,f)))
c=np.asarray(current,dtype='<f4');f=np.asarray(future,dtype='<f4')
with (B/'cases.bin').open('xb') as out:out.write(struct.pack('<I',len(groups)));out.write(c.tobytes());out.write(f.tobytes())
a.write_json(B/'groups.json',groups);a.write_json(B/'oracle.json',pred);a.write_json(A/'calibration.json',calibrate())
# Four exposed parent geometries are development diagnostics, never confirmation units.
parent=R/'build/crow-ccd-translation-build-01'
raw=(parent/'cases.bin').read_bytes();count=struct.unpack_from('<I',raw)[0]
pc=np.frombuffer(raw,dtype='<f4',offset=4,count=count*40).reshape(count,2,5,4)
pf=np.frombuffer(raw,dtype='<f4',offset=4+count*160,count=count*40).reshape(count,2,5,4)
parent_analysis=json.loads((parent/'analysis-full.json').read_text())
indices=sorted(set(x['record']//2 for x in parent_analysis['unsafe_successful_no_impact_records']))
assert len(indices)==4
with (B/'observed-misses.bin').open('xb') as out:
    out.write(struct.pack('<I',len(indices)));out.write(pc[indices].tobytes());out.write(pf[indices].tobytes())
queries=[];reference=[]
for i in indices:
    prediction=cast(*local_segment(pc[i],pf[i]));lo=prediction['first_alpha'];mid=prediction['minimum_alpha']
    queries.append([0.,lo,(lo+mid)/2,mid]);reference.append({'parent_geometry_index':i,**prediction})
np.asarray(queries,dtype='<f4').tofile(B/'query-alpha.f32')
a.write_json(B/'query-reference.json',reference)
inputs={str(p.relative_to(R)):{'sha256':a.file_hash(p),'bytes':p.stat().st_size} for p in B.rglob('*') if p.is_file() and p.suffix not in ('.stdout','.stderr') and '__pycache__' not in p.parts}
plan={'purpose':'engineering exploration; exact failure diagnosis plus unused synthetic conditions, no biological claim',
 'source_commit':a.git('rev-parse','HEAD').decode().strip(),'source_path':'src/metal/Collision.metal','candidate_sha256':a.file_hash(B/'Collision.candidate.metal'),
 'parent':{'study':'artifacts/crow-rl-ccd-translation-20261108','verdict':'contradicted: eight unsafe no-impact records, four grazing geometries'},
 'revision':'Retain separating-plane closure, remove its unsafe fixed time jump. Source and condition set changed explicitly; previous source/results retained. Confirm proposed floor mechanism with sixteen native query measurements at reference-derived times in the four prior missed geometries.',
 'question':'Can a translating fixed-orientation sphere-box pair advance from its separating plane without treating tangential motion or sphere spin as normal closing motion?',
 'model':'Candidate changes CA denominator to -dot(query.normal, relative endpoint displacement) only for analytic sphere-box with literal box-quaternion endpoint equality. Nonpositive closure certifies separation on that plane. Existing initial-overlap/stationary/speculative guards, distance queries, tolerance, sixteen-iteration budget and rotating-box path unchanged. For the eligible translating primitive only, remove the fixed ten-microsecond advancement floor and use nextafter(alpha,2) solely for representable-alpha progress.',
 'prediction':{'parent_missed_geometries_native_minimum_query_below_tolerance_by_at_least_0_5_um':True,'native_minimum_query_vs_FP64_reference_max_absolute_error_m':2.5e-7,'all_four_captured_pairs_resolve':True,'all_257_native_distance_query_records_per_capture_byte_exact':True,
    'zero_successful_unprotected_no_impact_records_for_FP64_reference_crossings_deeper_than_0_5_um':True,
    'initial_overlap_stationary_rotating_baseline_records_byte_exact':True,
    'all_eight_exposed_grazing_pair_order_records_become_impacts':True,'candidate_repetition_byte_exact':True,'eligible_unambiguous_native_success_fraction_at_least':.99,
    'eligible_unresolved_count_reduction_at_least_fraction':.50},
 'experimental_units':'4096 distinct synthetic projected geometries, both pair orders (8192 records); four retained real failing sweeps are development replication. Frames and pair orders are not independent biological replicates.',
 'seed':2651521901,'unused_followup_seed':2651522901,'inputs':inputs,
 'oracle':'Independent continuous FP64 point-to-AABB piecewise quadratic cast. Known face/edge/corner first contact, tangent null and initial overlap calibrated before native trials. 0.5 micrometre ambiguity band acknowledges float32 pose interpolation; all grey cases and records retained.',
 'validity':['Actual native mr_world_resolve_ccd from baseline/candidate shaders, no replacement host physics.','Native distance query bytes unchanged on exact captures.','Every missing, rejected or mismatched control blocks promotion.','No claim of general rotating-body correctness or physical beak capability from synthetic grid.'],
 'analysis':'Retain every raw record; compare flags/status, qualified reference crossings and pair order. Initial overlap, literal stationary and any rotating box are exact byte controls. Do not count a typed unresolved record as unsafe no-impact; it still fails success target. Any unsafe no-impact contradicts the repair.',
 'budget':{'native_grid_executions':4,'captured_replays':4,'native_parent_grazing_query_values':16,'timeout_seconds_each':180,'raw_max_bytes':33554432,
    'stop':'One baseline, candidate and candidate repeat on unused grid, one candidate regression over four exposed parent geometries, four candidate captured replays, and sixteen native read-only query values in the exposed geometries. No retries, tolerance/budget changes, learning run or production replacement. Further revisions require new sealed plan and unused conditions.'}}
a.write_json(A/'plan.json',plan)
for name in ('prepare.py','oracle.py','run.py'):
    shutil.copy2(B/name,A/name)
print(json.dumps({'cases':len(groups),'plan_sha256':a.file_hash(A/'plan.json'),'calibration':calibrate()}))
