from pathlib import Path
import json,hashlib,shutil,difflib,subprocess,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');base=q/'snapshots/source-true-residual-native-reporting-243-001';s=q/'snapshots/source-strict-refined-contact-observers-241-003';d=q/'analysis/source241-observer-composition-003';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();s.mkdir();d.mkdir()
f=json.loads((base/'frozen-source.json').read_text());cfg=json.loads((base/'configure-support-source.json').read_text())
for n,h in {**cfg['files'],**f['source_sha256']}.items():
 assert sha(base/n)==h,n;p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
bind=lambda p:json.loads((p/'frozen-source.json').read_text())['source_sha256']
a=q/'snapshots/source-integrated-configure-support-219-001';b=q/'snapshots/source-first-failure-dispatch-224-001';c=q/'snapshots/source-first-failure-dispatch-233-001';e=q/'snapshots/source-deformable-contact-trial-trace-236-001';z=q/'snapshots/source-deformable-contact-trial-trace-236-002'
eb=json.loads((q/'analysis/deformable-contact-trial-alpha-trace-236-001/source236-overlay.json').read_text())['candidate_source_bindings']
zb=json.loads((q/'analysis/deformable-contact-trial-alpha-trace-236-002/build-source236-001/source-build.json').read_text())['source_sha256']
stages=[('source224',a,b,bind(a),bind(b)),('source233',b,c,bind(b),bind(c)),('source236001',c,e,bind(c),eb),('source236002',e,z,eb,zb)]
results=[];added=set()
for label,parent,child,pm,cm in stages:
 names=sorted(n for n,h in cm.items() if pm.get(n)!=h)
 patch=''
 for n in names:
  assert sha(child/n)==cm[n],n
  before=(parent/n).read_text().splitlines(keepends=True) if n in pm else []
  if n in pm:assert sha(parent/n)==pm[n],n
  after=(child/n).read_text().splitlines(keepends=True)
  patch+=''.join(difflib.unified_diff(before,after,fromfile='a/'+n if n in pm else '/dev/null',tofile='b/'+n))
  if n not in pm:added.add(n)
 pp=d/(label+'.patch');pp.write_text(patch)
 exclusions=['--exclude=matter/tools/cardboard_probe.mm','--exclude=matter/tools/test_cardboard_solver_environment.py'] if label=='source236001' else ['--exclude=matter/src/runtime.mm'] if label=='source236002' else []
 check=subprocess.run(['git','apply','--check',*exclusions,str(pp)],cwd=s,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 (d/(label+'.check.stdout')).write_bytes(check.stdout);(d/(label+'.check.stderr')).write_bytes(check.stderr)
 results.append({'stage':label,'changed':names,'patch_sha256':sha(pp),'check_exit_code':check.returncode})
 (d/'stage-progress.json').write_text(json.dumps(results,indent=2)+'\n')
 assert check.returncode==0,check.stderr.decode()
 p=subprocess.run(['git','apply',*exclusions,str(pp)],cwd=s,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 assert p.returncode==0,p.stderr.decode()
 if label=='source236001':
  n='matter/tools/cardboard_probe.mm';p=s/n;txt=p.read_text()
  assert txt.count('std::array<std::string_view, 53u> kSource161DiagnosticEnvironmentKeys{')==1
  txt=txt.replace('std::array<std::string_view, 53u> kSource161DiagnosticEnvironmentKeys{','std::array<std::string_view, 54u> kSource161DiagnosticEnvironmentKeys{')
  needle='    "NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT",'
  assert txt.count(needle)==1;txt=txt.replace(needle,needle+'\n    "NM_FEM_DEFORMABLE_LIMIT_TRACE_SLOT",')
  assert txt.count('[(NSArray*)keysValue count] <= 53u')==1
  txt=txt.replace('[(NSArray*)keysValue count] <= 53u','[(NSArray*)keysValue count] <= 54u')
  needle='    const bool contractHasStrictInnerKey = keys.count >= 53u;'
  assert txt.count(needle)==1
  txt=txt.replace(needle,'''    const bool contractHasStrictInnerKey =
        [keys containsObject:@"NM_FEM_STRICT_INNER_FORCING"];
    const bool contractHasTrialTraceSlotKey =
        [keys containsObject:@"NM_FEM_DEFORMABLE_LIMIT_TRACE_SLOT"];
    const NSUInteger expectedRegistryCount = 49u +
        NSUInteger(contractHasWarmStartKey) +
        NSUInteger(contractHasIncidenceSortKey) +
        NSUInteger(contractHasDampingKey) +
        NSUInteger(contractHasStrictInnerKey) +
        NSUInteger(contractHasTrialTraceSlotKey);
    require(keys.count == expectedRegistryCount,
            "source-bound diagnostic registry omits or adds an unregistered key");''')
  needle='''            if (expectedKey == "NM_FEM_STRICT_INNER_FORCING" &&
                !contractHasStrictInnerKey)
                continue;'''
  assert txt.count(needle)==1
  txt=txt.replace(needle,needle+'''
            if (expectedKey == "NM_FEM_DEFORMABLE_LIMIT_TRACE_SLOT" &&
                !contractHasTrialTraceSlotKey)
                continue;''')
  txt=txt.replace('exact 49/50/51/52/53-key source contract','exact 49/50/51/52/53/54-key source contract')
  p.write_text(txt)
  n='matter/tools/test_cardboard_solver_environment.py';txt=(child/n).read_text()
  assert txt.count('self.assertEqual(len(env.NATIVE_ENV_KEYS), 53)')==1
  (s/n).write_text(txt.replace('self.assertEqual(len(env.NATIVE_ENV_KEYS), 53)','self.assertEqual(len(env.NATIVE_ENV_KEYS), 54)'))
  results[-1]['manual_resolution']={'files':['matter/tools/cardboard_probe.mm','matter/tools/test_cardboard_solver_environment.py'],'reason':'Strict inner-forcing and trial-trace branches each extended52-key registry independently. Compose54 keys; validate exact filtered registry size and preserve both legitimate53-key predecessor variants.'}
 if label=='source236002':
  p=s/'matter/src/runtime.mm';txt=p.read_text();needle='#include "numi/matter/full_contact_armijo_admission.hpp"'
  assert txt.count(needle)==1
  txt=txt.replace(needle,needle+'\n#include "numi/matter/deformable_limit_trace_json.hpp"')
  before='''                            line << ",\"barrier\":[";
                            capturedAppendFloat(line, record.barrier.x); line << ',';
                            capturedAppendFloat(line, record.barrier.y); line << ',';
                            capturedAppendFloat(line, record.barrier.z); line << ',';
                            capturedAppendFloat(line, record.barrier.w);'''
  after='''                            detail::appendDeformableLimitTraceBarrier(
                                line, record.barrier.x, record.barrier.y,
                                record.barrier.z, record.barrier.w,
                                capturedAppendFloat);'''
  assert txt.count(before)==1,repr(before)
  txt=txt.replace(before,after);p.write_text(txt)
  results[-1]['manual_resolution']={'files':['matter/src/runtime.mm'],'reason':'Retain composed arena header includes while adding trace serializer helper; replace exactly the original5-line malformed barrier emitter with the tested helper.'}
for n in list(f['source_sha256'])+list(added):
 if (s/n).is_file():f['source_sha256'][n]=sha(s/n)
for n in cfg['files']:cfg['files'][n]=sha(s/n)
f['source241_composition']={'base_snapshot':str(base),'base_frozen_sha256':sha(base/'frozen-source.json'),'stages':results,'physical_operator_changed':False,'diagnostics_default_off':True,'native_composition_qualification':False}
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n');(s/'configure-support-source.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
(d/'report.json').write_text(json.dumps({'schema':'numi.cardboard.source241-observer-composition.v1','source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_count':len(f['source_sha256']),'stages':results,'patches_applied':True,'all_inputs_unchanged':all(sha(s/n)==h for n,h in {**cfg['files'],**f['source_sha256']}.items()),'gpu_execution':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
print((d/'report.json').read_text(),flush=True)
