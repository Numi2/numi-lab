from pathlib import Path
import hashlib,json,os,subprocess,time,csv
R=Path.cwd();B=R/'build/crow-bounded-drive-integration-build-01';A=R/'artifacts/crow-rl-bounded-drive-crow-20261108/terminal';T=R/'.numi/retained-runtime/bounded-drive-twofold-20261108'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
p=json.loads((A/'plan.json').read_text())
for path,h in p['inputs'].items():assert sha(R/path)==h,path
assert not subprocess.check_output(['git','status','--porcelain'])
env=os.environ.copy();env.update(DYLD_LIBRARY_PATH=str(T),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(T/'metalrobo/MetalRobo.metallib'));env.pop('METALROBO_BOUNDED_FAILURE_CAPTURE',None)
cmd=[str(B/'fixture-terminal'),str(B/'terminal-crow32'),'0','1','32'];start=time.monotonic()
with (B/'terminal-crow32.stdout').open('xb') as out,(B/'terminal-crow32.stderr').open('xb') as err:code=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=300).returncode
report={'command':cmd,'exit_code':code,'seconds':time.monotonic()-start,'sealed_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'policy_or_profile_promoted':False}
write(B/'terminal-execution.json',report)
assert code==1 and (B/'terminal-crow32/terminal.json').exists(),report
report['terminal']=json.loads((B/'terminal-crow32/terminal.json').read_text());report['prefix_trace_byte_exact']=(B/'terminal-crow32/trace.tsv').read_bytes()==(B/'twofold-crow32/trace.tsv').read_bytes()
assert report['prefix_trace_byte_exact']
rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((B/'terminal-crow32/trace.tsv').open(),delimiter='\t')]
report['accepted_control_transitions']=len(rows)+report['terminal']['accepted'];report['native_rejections']=report['terminal']['failed'];report['maximum_observed_penetration_m']=max(0.,-min(min(r['upper_gap_m'],r['lower_gap_m']) for r in rows))
report['verdict']='physical_task_termination_reproduced_with_terminal_evidence';report['limits']='Same exposed trajectory, not an independent physical replicate. The former generic runner label inconclusive_crow_native_failure did not distinguish this accepted task termination. This is a physical failure of the fixed fixture, not a GPU motor certificate rejection.'
write(A/'results.json',report);write(B/'terminal-analysis.json',report);print(json.dumps(report),flush=True)
