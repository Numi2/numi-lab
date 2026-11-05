from pathlib import Path
import importlib.util,json,time,hashlib,resource
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
spec=importlib.util.spec_from_file_location('owner',W/'compare_remeshed_native_310.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
original=owner.capture;cache={}
def cached(path):
 key=str(path.resolve())
 if key not in cache:cache[key]=original(path)
 return cache[key]
owner.capture=cached
summary=[]
for name,ref,cand in [
 ('ccd-control-335-vs-324','baseline-repeat-native-324','matched-ccd-baseline-335'),
 ('interior-334-vs-matched-335','matched-ccd-baseline-335','compensated-interior-native-334')]:
 start=time.time()
 result=owner.compare(W/ref/'install-motion.stdout.log.gz',W/cand/'install-motion.stdout.log.gz')
 out=W/(name+'-comparison.json');out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 sf=[k for k,v in result['regional_strain'].items() if v['relative_change']['volume_weighted_strain_norm'] is None or v['relative_change']['volume_weighted_strain_norm']>.05]
 ff=[(k,c) for k,v in result['regional_force_channels'].items() for c,r in v.items() if r['relative_resultant_difference_over_reference_nodal_l1'] is None or r['relative_resultant_difference_over_reference_nodal_l1']>.01]
 row={'name':name,'analysis_wall_s':time.time()-start,'reference_body':result['reference']['body'],'candidate_body':result['candidate']['body'],'body_wall_reduction_fraction':result['body_wall_reduction_fraction'],'body_throughput_ratio':result['body_throughput_ratio'],'strain_metric_failures':len(sf),'force_channel_failures':len(ff),'reference_active_work':result['reference']['active_work'],'candidate_active_work':result['candidate']['active_work'],'all_coupled_fields_identical':all(v['max_abs_change']==0 for v in result['coupled_trace'].values()),'all_strain_aggregates_identical':all(v['reference']==v['candidate'] for v in result['regional_strain'].values()),'all_force_aggregates_identical':all(r['reference']==r['candidate'] for v in result['regional_force_channels'].values() for r in v.values()),'physical_fidelity_qualification':False}
 summary.append(row);print(json.dumps(row),flush=True)
(W/'matched-ccd-336-summary.json').write_text(json.dumps({'results':summary,'comparison_owner_sha256':hashlib.file_digest((W/'compare_remeshed_native_310.py').open('rb'),'sha256').hexdigest(),'cached_loaded_captures':len(cache),'maximum_analysis_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2)+'\n')
