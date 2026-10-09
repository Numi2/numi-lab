#!/usr/bin/env python3
import hashlib,json,pathlib,re,statistics
ROOT=pathlib.Path('/Users/n/numi-human-retained-delivery-20261009')
RUNS={'flat_reference':ROOT/'native-bed-disabled-parity-1197-attempt002','contoured_bed':ROOT/'native-contoured-bed-smoke-1198-attempt002'}
OUT=ROOT/'native-bed-profile-comparison-1198'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def readj(p):return json.load(open(p))
def parse_values(line):
 d={}
 for k,v in re.findall(r'([A-Za-z0-9_]+)=([^ ]+)',line):
  try:d[k]=float(v)
  except ValueError:d[k]=v
 return d
def profiles(log,prefix):
 return [parse_values(x[len(prefix):]) for x in log.splitlines() if x.startswith(prefix)]
def stats(rows,key):
 x=[float(r[key]) for r in rows if isinstance(r.get(key),(int,float))]
 if not x:return {'count':0}
 return {'count':len(x),'sum_ms':sum(x),'mean_ms':statistics.mean(x),'median_ms':statistics.median(x),'p95_ms':statistics.quantiles(x,n=20)[18],'max_ms':max(x),'min_ms':min(x)}
def percent(a,b):return 100*(b-a)/a if a else None
def envcmd(d):
 arr=readj(d/'run-declaration.json')['argv'];env={};start=arr.index(next(x for x in arr if x.endswith('/tools/numi')))
 for x in arr[2:start]:
  if '=' in x:
   k,v=x.split('=',1);env[k]=v
 return env,arr[start:]
def normalized_native_argv(argv, remove_bed_option=False):
 out=[];i=0
 while i<len(argv):
  x=argv[i]
  if remove_bed_option and x=='--resting-bed-surface' and i+1<len(argv):
   i+=2;continue
  if x in ('--body-scene','--resting-bed-surface','--resting-movie') and i+1<len(argv):
   out.extend([x,'<SCENE>' if x=='--body-scene' else '<BED>' if x=='--resting-bed-surface' else '<MOVIE>']);i+=2;continue
  if x.endswith('/native-run'):
   out.append('<OUTPUT>')
  else:out.append(x)
  i+=1
 return out

data={}; pins={}
for label,run in RUNS.items():
 native=run/'native-run'; meta=readj(native/'run-metadata.json'); src=readj(run/'runtime-source-at-launch.json')
 logpath=native/'native.log'; text=logpath.read_text(errors='replace')
 native_rows=profiles(text,'resting_native_profile ')
 render=profiles(text,'resting_render_profile ');present=profiles(text,'resting_present_profile ')
 regular=[x for x in render if x.get('geometry_export')==0]; exports=[x for x in render if x.get('geometry_export')==1]
 obs_match=re.search(r'^resting_integrated_observer_profile (.*)$',text,re.M)
 throughput_match=re.search(r'^resting_integrated_throughput (.*)$',text,re.M)
 body_match=re.search(r'^resting_integrated_body=completed simulated_s=([^ ]+) wall_s=([^ ]+) real_time_factor=([^ ]+)',text,re.M)
 observer=parse_values(obs_match.group(1)) if obs_match else {}
 throughput=parse_values(throughput_match.group(1)) if throughput_match else {}
 body=[float(x) for x in body_match.groups()] if body_match else None
 data[label]={
  'run_path':str(run),
  'execution':readj(run/'execution.json'),
  'run_metadata_summary':{k:meta.get(k) for k in ('exit_code','wall_seconds','source_files_changed_during_run','loaded_metal_runtime')},
  'runtime':{'base_revision':src['base_revision'],'runtime_patch_sha256':src['patch_sha256'],'native_source_file_hashes':src['runtime_source_files'],'native_binary_sha256':meta['asset_sha256'].get(meta['argv'][0]),'loaded_metalrobo_sha256':meta['loaded_metal_runtime']['expected_sha256'],'profile_rows':{'native':len(native_rows),'render':len(render),'present':len(present)}},
  'assets':meta['asset_sha256'],
  'profile':{'native_command_ms':stats(native_rows,'command_ms'),'native_gpu_ms':stats(native_rows,'gpu_ms'),'native_host_copy_ms':stats(native_rows,'host_copy_ms'),'render_regular_count':len(regular),'render_regular_wall_ms':stats(regular,'total_render_wall_ms'),'render_regular_command_ms':stats(regular,'command_wall_ms'),'render_regular_gpu_ms':stats(regular,'gpu_ms'),'full_geometry_exports':len(exports),'geometry_vertices':sorted(set(x['vertices'] for x in exports)),'geometry_triangles':sorted(set(x['triangles'] for x in exports)),'geometry_export_audit_ms':stats(exports,'audit_export_wall_ms'),'geometry_export_total_wall_ms':stats(exports,'total_render_wall_ms'),'presentation_count':len(present),'presentation_wall_ms':stats(present,'total_present_wall_ms'),'observer':observer,'integrated_throughput':throughput,'integrated_body':{'simulated_s':body[0],'wall_s':body[1],'real_time_factor':body[2]} if body else None}
 }
 for rel in ('execution.json','run-declaration.json','runtime-source-at-launch.json','runtime-source-at-launch.patch','native-run/run-metadata.json','native-run/invocation.json','native-run/native.log','owner-stdout.log'):
  p=run/rel;pins[str(p)]={'sha256':sha(p),'size_bytes':p.stat().st_size}

base=data['flat_reference'];bed=data['contoured_bed']
def delta(path):
 a=base;b=bed
 for k in path:a=a[k];b=b[k]
 return {'flat':a,'contoured_bed':b,'absolute_delta':b-a,'relative_change_percent':percent(a,b)}
comp={k:delta(k.split('.')) for k in (
 'execution.wall_seconds','profile.integrated_body.wall_s','profile.integrated_body.real_time_factor',
 'profile.native_command_ms.mean_ms','profile.native_gpu_ms.mean_ms','profile.native_host_copy_ms.mean_ms',
 'profile.render_regular_wall_ms.mean_ms','profile.render_regular_gpu_ms.mean_ms',
 'profile.geometry_export_audit_ms.mean_ms','profile.geometry_export_total_wall_ms.mean_ms','profile.presentation_wall_ms.mean_ms')}
obsdelta={}
for k in sorted(set(base['profile']['observer'])|set(bed['profile']['observer'])):
 a=base['profile']['observer'].get(k);b=bed['profile']['observer'].get(k)
 if isinstance(a,(int,float)) and isinstance(b,(int,float)):obsdelta[k]={'flat':a,'contoured_bed':b,'delta':b-a,'relative_change_percent':percent(a,b)}
env0,declcmd0=envcmd(RUNS['flat_reference']);env1,declcmd1=envcmd(RUNS['contoured_bed'])
launchdiff={k:[env0.get(k),env1.get(k)] for k in sorted(set(env0)|set(env1)) if env0.get(k)!=env1.get(k)}
meta0=readj(RUNS['flat_reference']/'native-run/run-metadata.json');meta1=readj(RUNS['contoured_bed']/'native-run/run-metadata.json')
native_argv0=meta0['argv'];native_argv1=meta1['argv']
normalized_argv_equal=normalized_native_argv(native_argv0)==normalized_native_argv(native_argv1,remove_bed_option=True)
added_native_flags=sorted(set(x for x in native_argv1 if x.startswith('--'))-set(x for x in native_argv0 if x.startswith('--')))
removed_native_flags=sorted(set(x for x in native_argv0 if x.startswith('--'))-set(x for x in native_argv1 if x.startswith('--')))
patch0=(RUNS['flat_reference']/'runtime-source-at-launch.patch').read_text().splitlines()
patch1=(RUNS['contoured_bed']/'runtime-source-at-launch.patch').read_text().splitlines()
import difflib
source_patch_diff=[x for x in difflib.unified_diff(patch0,patch1,n=0) if x.startswith(('---','+++','@@','+','-'))]
report={'schema':'numi.native-contoured-bed-profile-comparison.v1','scope':'Retained complete 20 s runs; measured run-to-run bed-mode profile difference, not a repeated controlled benchmark and not single-kernel attribution. No GPU or source edits.','comparison':comp,'observer_phase_deltas':obsdelta,'runs':data,'launch_comparison':{'same_base_revision':base['runtime']['base_revision']==bed['runtime']['base_revision'],'base_revision':base['runtime']['base_revision'],'runtime_patch_sha256':{'flat':base['runtime']['runtime_patch_sha256'],'contoured':bed['runtime']['runtime_patch_sha256']},'native_binary_hashes_differ':base['runtime']['native_binary_sha256']!=bed['runtime']['native_binary_sha256'],'same_loaded_metalrobo_hash':base['runtime']['loaded_metalrobo_sha256']==bed['runtime']['loaded_metalrobo_sha256'],'native_command_equal_after_normalizing_output_scene_movie_paths_and_removing_declared_bed_option':normalized_argv_equal,'added_native_flags':added_native_flags,'removed_native_flags':removed_native_flags,'environment_differences':launchdiff,'environment_differences_after_normalizing_failure_receipt_path':{k:v for k,v in launchdiff.items() if k!='NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT'},'source_patch_diff_stat_lines':len(source_patch_diff),'source_patch_diff_excerpt':source_patch_diff[:20],'contoured_surface_flag_added':'--resting-bed-surface' in added_native_flags,'source_delta_summary':'Both launch snapshots share the same base revision and match in native argv after normalizing run-output/movie paths and removing the declared --resting-bed-surface option. Their retained source patches differ in the contoured-bed mesh shading tangent construction (attempt002 contains the orthonormal tangent fix) and line-offset metadata. The native executable hashes differ; loaded MetalRobo dylib and shader-library pins match. The contoured invocation adds the bed manifest and --resting-bed-surface; the failure-receipt environment value also points to each run own output directory.'},'instrumentation_limit':'1250 per-segment native command/GPU/host-copy samples, 315 render/present samples, and aggregate accepted-observer callback phase totals are available. The logs do not separately time the bed query/contact-plane update kernel, so GPU delta is mode-associated, not per-kernel attribution.','qualification_limit':'1198 is a 20 s profile only; later anatomy audit reported 3,727 skin-target crossings across 26 surfaces and zero self-intersections. This comparison does not qualify anatomy or long-horizon timing.','input_file_hashes':pins}
OUT.mkdir(parents=True,exist_ok=True)
p=OUT/'profile-comparison.json';p.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'report_path':str(p),'report_sha256':sha(p),'script_sha256':sha(__file__),'comparison':comp,'observer_phase_deltas':obsdelta},indent=2))
