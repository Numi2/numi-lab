"""Bound completed native tool trajectories before comparing numerical barrier factors."""
import csv,hashlib,json,math,os,sys
from pathlib import Path
root=Path(__file__).resolve().parent
repo=Path(os.environ.get('NUMI_CARDBOARD_REPO','/Users/home/numi-cardboard-20261006'))
sys.path.insert(0,str(repo/'matter/tools'))
from cardboard_tooling_analysis import analyze_run
names=[f'barrier50-scale{s}-tight-001' for s in [1,2,4]]
report={'schema':'numi.cardboard.tight-barrier-comparison.v1','physical_validation':False,'runs':{},'paired_status':'inconclusive'}
normalized=[];bindings=[];curves={}
for name in names:
 p=root/name
 if not (root/'receipts'/name/'receipt.json').exists():
  report['runs'][name]={'status':'not_completed'};continue
 analysis=analyze_run(p);manifest=json.loads((p/'manifest.json').read_text());receipt=json.loads((root/'receipts'/name/'receipt.json').read_text())
 obs=[r for r in csv.DictReader((p/'observations.csv').open()) if r['environment']=='0'];tool={int(r['step']):r for r in csv.DictReader((p/'tool-observations.csv').open()) if r['environment']=='0'}
 accepted=[r for r in obs if r['status_code']=='0' and r['step_accepted']=='1']
 rows=[tool[int(r['step'])] for r in accepted]
 report['runs'][name]={'status':analysis['result_status'],'verdict':analysis['verdict'],'accepted_steps':analysis['accepted_steps'],'scale':manifest['tooling']['prescribed_tool_barrier_stiffness_scale'],'minimum_accepted_punch_gap_m':min((float(r['min_end_punch_node_gap_m']) for r in rows),default=None),'peak_accepted_punch_force_N':max((abs(float(r['punch_force_z_N'])) for r in rows),default=None),'maximum_accepted_free_force_imbalance_l2_N':max((float(r['free_force_imbalance_l2_N']) for r in accepted),default=None),'inputs_unchanged':receipt['inputs_unchanged'],'input_sha256':analysis['sha256_bindings']['artifacts']}
 manifest.pop('compiled_world_fingerprint');manifest['tooling'].pop('prescribed_tool_barrier_stiffness_scale');normalized.append(manifest);bindings.append(receipt['bindings'])
 curves[name]=[(int(r['step']),r['phase'],float(r['time_s']),float(tool[int(r['step'])]['punch_force_z_N'])) for r in obs]
if all(report['runs'][n].get('verdict')=='instrument_checks_passed' for n in names):
 assert all(x==normalized[0] for x in normalized),'unexpected manifest differences'
 assert all(x==bindings[0] for x in bindings),'native inputs differ'
 assert all(report['runs'][n]['inputs_unchanged'] for n in names)
 report['paired_status']='all_complete_same_native_inputs'
 peaks=[report['runs'][n]['peak_accepted_punch_force_N'] for n in names]
 spread=(max(peaks)-min(peaks))/max(peaks)
 report['peak_force_relative_spread']={'value':spread,'declared_threshold':.1,'within_threshold':spread<=.1,'denominator':'maximum of the three peak magnitudes'}
 report['force_curve_comparisons']={}
 for i,a in enumerate(names):
  for b in names[i+1:]:
   assert [x[:3] for x in curves[a]]==[x[:3] for x in curves[b]],'time/phase grids differ'
   phases={}
   for phase in ['loading','hold','unloading','relaxation']:
    av=[x[3] for x in curves[a] if x[1]==phase];bv=[x[3] for x in curves[b] if x[1]==phase]
    rms=lambda v:math.sqrt(sum(x*x for x in v)/len(v))
    if not av:continue
    delta=rms([x-y for x,y in zip(av,bv)]);denom=max(rms(av),rms(bv))
    phases[phase]={'rms_difference_N':delta,'relative_rms_difference':delta/denom if denom else None,'samples':len(av)}
   report['force_curve_comparisons'][a+' vs '+b]=phases
report['analyzer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
report['tooling_analyzer_sha256']=hashlib.sha256((repo/'matter/tools/cardboard_tooling_analysis.py').read_bytes()).hexdigest()
out=root/'tight-barrier-comparison.json'
if out.exists():raise SystemExit('refusing to overwrite retained report')
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
