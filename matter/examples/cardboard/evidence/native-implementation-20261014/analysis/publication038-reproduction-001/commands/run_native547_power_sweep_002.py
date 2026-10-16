from pathlib import Path
import json,hashlib,subprocess,os,time,datetime,signal,sys
q=Path("/Users/n/cardboard-implementation-20261014-host-macmini");c=q/'native/critical23-power-sweep-547';p=q/'analysis/native547-critical23-sweep-input-002'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=json.loads((q/'native/critical23-material-replay-541/command.json').read_text())['argv'];a[4]=str(p/'input.json');a[5]=sha(p/'input.json');a[6]=str(c/'output')
assert a[5]=='a71da761e88cbaca694e7c0cd6333511f46a0a36a12b76b5081b0d26309bb74e'
bindings={a[0]:'7764e4c20c2393d5ebfc3f100f29e851b9090e1999418aad86cbcaf69fb7e3c5',a[2]:a[3],a[4]:a[5],str(p/'stable-fp64-reference.jsonl'):'2893d2b988feae51fe6ecbe3a7a71ecb7a19d1f9b011a3957b52df54c4bd96e0',str(p/'pair-selection.json'):'36dfab1b02c443ef9ee77df2cada5e7eafa9abe3450c1d33208016357ceeb452'}
assert all(sha(n)==h for n,h in bindings.items())
assert sha(p/'admission-v2.stdout')=='9810c22bc4d6062df760e2847b1214cec0b6e3b8b5f53689c2c297e1381a07af'
audit=q/'analysis/native547-critical23-sweep-input-audit-008/report.json'
assert sha(audit)=='43d1cd7456c087efac12eb3a0af232953af417627904c55c6474466890d1437a'
ad=json.loads(audit.read_text());v2=[r for r in ad['results'] if r['label']=='corrected_547_v2'][0]
assert v2['zero_endpoint_dF']==0 and v2['wrong_or_zero_sign_endpoint_dF']==0 and v2['accepted_state_word_change_rows']==0 and v2['reference_center_mismatch_rows']==0 and v2['max_full_F_target_error']<2e-15
bindings[str(audit)]=sha(audit)
ps=subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True)
workers=[l for l in ps.splitlines() if any(x in l for x in ['numi-matter-','generic-batch','geometry-replay']) and not l.split()[1].startswith('Z')]
assert not workers,workers
assert not c.exists();c.mkdir()
command={'schema':'numi.cardboard.native547-command.v1','argv':a,'bindings':bindings,'source_capture_sha256':a[7],'power_range':[10,40],'expected_rows':1472,'question':'Does the exact production material package admit a resolved finite-difference range at any adjacent representable powers, with unchanged accepted history and projection budget?','prediction':'A broad sweep may expose a truncation-to-noise window, but no resolved derivative gate is assumed. FP32 under-resolution remains inconclusive.','all_541_failures_preserved':True,'corrected_input_package':str(p),'superseded541_FD_endpoints_invalid_from_mutating_builder':True,'superseded547_001_not_run':True}
(c/'command.json').write_text(json.dumps(command,indent=2)+'\n')
start=time.monotonic();timed=False;err=None;code=None
try:
 with (c/'stdout.log').open('xb') as out,(c/'stderr.log').open('xb') as er:
  proc=subprocess.Popen(a,stdout=out,stderr=er,env={'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','HOME':'/Users/n'},start_new_session=True)
  try:code=proc.wait(timeout=120)
  except subprocess.TimeoutExpired:
   timed=True;os.killpg(proc.pid,signal.SIGTERM)
   try:code=proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait(timeout=10)
except BaseException as ex:err=repr(ex)
r={'schema':'numi.cardboard.native547-receipt.v1','exit_code':code,'timed_out':timed,'error':err,'wall_seconds':time.monotonic()-start,'physical_transactions':0,'bindings_unchanged':all(sha(n)==h for n,h in bindings.items()),'stdout_sha256':sha(c/'stdout.log'),'stderr_sha256':sha(c/'stderr.log'),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'qualification':'Material replay only; all derivative gates require separate adjudication.'}
(c/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
sys.exit(int(code!=0 or timed or bool(err) or not r['bindings_unchanged']))
