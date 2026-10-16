from pathlib import Path
import json,shutil,hashlib,subprocess,os,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');base=q/'snapshots/source-refined-regional-graph-240-004';s=q/'snapshots/source-true-residual-native-reporting-243-001';d=q/'analysis/source243-native-reporting-001';b=q/'build-source243-native-reporting-001';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((base/'frozen-source.json').read_text());cfg=json.loads((base/'configure-support-source.json').read_text());s.mkdir();d.mkdir()
for n,h in {**cfg['files'],**f['source_sha256']}.items():
 assert sha(base/n)==h,n;p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
n='matter/tools/fgmres_true_residual_native_check.mm';p=s/n;t=p.read_text()
anchor='''    }
}

} // namespace'''
assert t.count(anchor)==1
replacement='''    }
    std::cout << "true_residual_measurement_nodes=" << nodeCount
              << " unknown_rows=" << unknownCount
              << " constrained_mechanical_rows=" << fixedMechanical
              << " constrained_field_rows=" << fixedField
              << " measurement_only=true full_FEM_operator_qualified=false"
              << " result=PASS" << '\\n';
}

} // namespace'''
t=t.replace(anchor,replacement)
old='''            std::cout << "large_row_fixture=4096,8192 measurement_only=true full_FEM_operator_at_N8192=false\\\\n";
'''
assert t.count(old)==1
t=t.replace(old,'')
assert 'large_row_fixture=' not in t and 'true_residual_measurement_nodes=' in t
p.write_text(t);f['source_sha256'][n]=sha(p)
f['source243_reporting']={'base_frozen_sha256':sha(base/'frozen-source.json'),'delta':[n],'physical_operator_changed':False,'old_native581_stdout_preserved':True,'purpose':'Emit one completion record from each actual measurement helper with its passed dimensions and masks; remove stale hardcoded node list and literal backslash-n.'}
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n');(s/'configure-support-source.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
a=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--cmake-source-relative','.','--jobs','2','--target','numi-matter-fgmres-true-residual-native-check']
with (d/'build.log').open('xb') as out:p=subprocess.run(a,cwd=s,stdout=out,stderr=subprocess.STDOUT,env=env)
r={'schema':'numi.cardboard.source243-native-reporting.v1','source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'build_argv':a,'build_exit_code':p.returncode,'all_pass':p.returncode==0,'all_inputs_unchanged':all(sha(s/n)==h for n,h in {**cfg['files'],**f['source_sha256']}.items()),'gpu_execution':False,'operator_or_gate_change':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(int(not r['all_pass']))
