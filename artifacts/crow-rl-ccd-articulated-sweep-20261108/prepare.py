from pathlib import Path
import importlib.util,json,os,shutil,subprocess
R=Path.cwd();B=R/'build/crow-ccd-articulated-sweep-build-01';A=R/'artifacts/crow-rl-ccd-articulated-sweep-20261108';P=R/'build/crow-ccd-contact-time-build-01'
assert not subprocess.check_output(['git','status','--porcelain'])
A.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('archive',R/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
entries=a.tree()
with a.Objects() as obj:m=json.loads(obj.read(entries['artifacts/crow-rl-ccd-event-status-20261108/payload/manifest.json'][1]))
f=next(f for f in m['files'] if f['path']=='build/crow-beak-hold-build-02/probe-with-runtime.mm')
with a.Objects() as obj,(B/'fixture-original.mm').open('xb') as out:
    for block in a.content(f,entries,obj):out.write(block)
s=(B/'fixture-original.mm').read_text().replace('#include <chrono>','#include <chrono>\n#include <cstdlib>')
for old,new in [('position.x+=(lane&1u)?.00025f:-.00025f;','position.x+=(lane&1u)?.00035f:-.00035f;'),
 ('position.z+=(lane&2u)?.00015f:-.00015f;','position.z+=(lane&2u)?.00020f:-.00020f;'),
 ('position.y=(lane&1u)?.00015f:-.00015f;','position.y=(lane&1u)?.00010f:-.00010f;')]:
 assert s.count(old)==1;s=s.replace(old,new)
needle='MetalWorldConfig config;config.enableResidentCheckpoints=true;'
assert s.count(needle)==1
s=s.replace(needle,'''MetalWorldConfig config;
    const char* selected=std::getenv("CROW_PROBE_METALLIB");
    require(selected&&std::filesystem::is_regular_file(selected),"Explicit CROW_PROBE_METALLIB required");
    config.metallibPath=selected;std::cerr<<"Crow probe metallib: "<<config.metallibPath<<'\\n';
    config.enableResidentCheckpoints=true;''')
needle='''        require(std::abs(q.back()-.23)<1e-7,"Initial jaw reference");''';assert s.count(needle)==1
s=s.replace(needle,needle+'''
        const unsigned head=body(m,"crow_head"),beak=body(m,"crow_lower_beak");unsigned upper=MR_INVALID_INDEX,lower=MR_INVALID_INDEX;
        for(unsigned i=0;i<m.shapes.size();++i){if(m.shapeNames[i]=="crow_head/upper_beak_collision")upper=i;if(m.shapes[i].bodyIndex==beak)lower=i;}
        require(upper!=MR_INVALID_INDEX&&lower!=MR_INVALID_INDEX,"Calibration jaw shapes");double minimum=1;
        for(unsigned lane=0;lane<4;++lane){mr_float4 p{.215f+((lane&1u)?.00035f:-.00035f),(lane&1u)?.00010f:-.00010f,.26685f+((lane&2u)?.00020f:-.00020f),1};
            minimum=std::min({minimum,gap(k[head],m.shapes[upper],p),gap(k[beak],m.shapes[lower],p)});}
        require(minimum>2e-5,"Unused poses initially overlap a jaw");
        std::cout<<"{\\"unused_pose_calibration\\":\\"passed\\",\\"minimum_gap_m\\":"<<std::setprecision(12)<<minimum<<"}\\n";
''')
(B/'fixture.mm').write_text(s)
s=(P/'Collision.candidate.metal').read_text()
needle='''        (source.flags & MR_SHAPE_FLAG_ENABLE_CCD) != 0u;'''
assert s.count(needle)==1
s=s.replace(needle,'''        (dispatch.ccdMode == MR_WORLD_CCD_HYBRID ||
         (source.flags & MR_SHAPE_FLAG_ENABLE_CCD) != 0u);''')
needle='''            speculativeMargin > 0.0f ? 1u : 0u;''';assert s.count(needle)==1
s=s.replace(needle,'''            as_type<uint>(speculativeMargin);''')
(B/'Collision.candidate.metal').write_text(s)
for name in ('cases-probe','cases.bin','candidate.bin'):
    shutil.copy2(P/name,B/('expected-grid.bin' if name=='candidate.bin' else name))
shutil.copy2(P/'candidate.metallib',B/'baseline.metallib')
link=json.loads((P/'link-command.json').read_text());link=[str(B/'candidate.air') if x==str(P/'candidate.air') else str(B/'candidate.metallib') if x==str(P/'candidate.metallib') else x for x in link]
a.write_json(B/'link-command.json',link);shutil.copy2(P/'unchanged-air-identities.json',B/'unchanged-air-identities.json')
commands=[]
def execute(name,cmd,env=None):
 with (B/(name+'.stdout')).open('xb') as out,(B/(name+'.stderr')).open('xb') as err:code=subprocess.run([str(x) for x in cmd],stdout=out,stderr=err,env=env,timeout=180).returncode
 commands.append({'name':name,'command':[str(x) for x in cmd],'exit_code':code});a.write_json(B/'prepare-execution.json',commands);assert code==0,(name,code)
execute('compile-projection',['clang++','-std=c++20','-O3','-fno-fast-math','-fobjc-arc','-I',R/'include',B/'projection.mm','-framework','Foundation','-framework','Metal','-o',B/'projection-probe'])
native=R/'.numi/retained-runtime/ccd-event-capture-20261108/candidate/libmetalrobo.dylib'
execute('compile-fixture',['clang++','-std=c++20','-O3','-fno-fast-math','-fobjc-arc','-I',R/'include',B/'fixture.mm',native,'-Wl,-rpath,@loader_path/runtime','-framework','Foundation','-o',B/'probe'])
for p,h in json.loads((B/'unchanged-air-identities.json').read_text()).items():assert a.file_hash(p)==h
execute('compile-candidate',['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',R/'include','-c',B/'Collision.candidate.metal','-o',B/'candidate.air'])
execute('link-candidate',link)
runtime=B/'runtime';(runtime/'metalrobo').mkdir(parents=True,exist_ok=False);shutil.copy2(native,runtime/'libmetalrobo.dylib');shutil.copy2(B/'candidate.metallib',runtime/'metalrobo/MetalRobo.metallib')
env=os.environ.copy();env['DYLD_LIBRARY_PATH']=str(runtime)
execute('calibrate-unused-poses',[B/'probe','--calibrate'],env)
inputs=[p for p in B.rglob('*') if p.is_file() and p.suffix not in ('.stdout','.stderr','.air') and '__pycache__' not in p.parts]
a.write_json(A/'plan.json',{'question':'Does authoritative hybrid articulated motion plus correctly encoded speculative margin remove the coarse clamp failure and preserve a converged physical gesture?',
 'source_base':a.git('rev-parse','HEAD').decode().strip(),'parent':'artifacts/crow-rl-ccd-contact-time-20261108',
 'model_revision':'Keep supporting-plane contact time candidate. In hybrid CCD, consume future FK for every articulated collider, because an unflagged collider can pair with a CCD sphere. In the speculative fast path, encode the actual float margin bits, matching the existing reader, rather than Boolean1. No tolerance, budget, contact geometry, actuation or solver changes.',
 'native_projection_controls':'128 prescribed distinct future poses; two identical shapes on the same articulation differing only in own CCD flag, under hybrid and speculative modes. These are deterministic kernel conditions, not physical/biological replicates.',
 'predictions':['Hybrid unflagged and flagged shapes have identical projected geometry and float margin, retaining their own source flag metadata.','Speculative unflagged fast path retains all geometry but corrects exactly its float margin bits to the known generalized-speed envelope.','Flagged future FK geometry remains byte exact against baseline.','Candidate projection repetition is byte exact.','Existing4096 contact-time records remain byte exact.'],
 'physical_conditions':'Four unused positions around(0.215,0,0.26685): x±0.00035, z±0.00020 in full2x2; y±0.00010 according to x. Same clamp/lift/wide-release actions, no attachments, food ledger or retinal policy.',
 'physical_gates':{'all_six_runs_and3840_transitions_accepted':True,'each_positive_lane_minimum_lift_64_95_m':.020,'maximum_slip_m':.005,'opposed_hold_samples_at_least':16,'maximum_penetration_before96_m':.0008,'release_drop_m':.030,'minimum_gap_ticks128_159_m':.001,'chunk1_vs8_final_q_v_scene':'byte exact','32vs64_max_paired_object_position_difference_all160ticks_m':.005,'controls':'Open, no-contact and zero-friction each fail the primary hold/lift. No-contact disables ground too.'},
 'trials':[{'name':'clamp','arm':0,'chunk':1,'substeps':32},{'name':'open','arm':1,'chunk':1,'substeps':32},{'name':'no-contact','arm':2,'chunk':1,'substeps':32},{'name':'zero-friction','arm':3,'chunk':1,'substeps':32},{'name':'partition','arm':0,'chunk':8,'substeps':32},{'name':'refinement','arm':0,'chunk':1,'substeps':64}],
 'inputs':{str(p.relative_to(R)):{'sha256':a.file_hash(p),'bytes':p.stat().st_size} for p in inputs},
 'budget':{'projection_executions':3,'grid_executions':1,'physical_executions':6,'timeout_seconds_each':180,'new_raw_max_bytes':33554432,'stop':'One execution per condition; failed projection controls stop before physical trials. Retain all missing/rejected/negative results, no retries or threshold changes. No production replacement or biological/general manipulation qualification from this trial alone.'}})
for name in ('prepare.py','run.py','projection.mm','fixture.mm'):shutil.copy2(B/name,A/name)
print(json.dumps({'plan_sha256':a.file_hash(A/'plan.json'),'calibration':(B/'calibrate-unused-poses.stdout').read_text()}),flush=True)
