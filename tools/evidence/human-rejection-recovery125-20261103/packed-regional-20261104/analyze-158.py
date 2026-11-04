from pathlib import Path
import csv,json,re,math,hashlib
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
a=D/'interpolated-armijo-native-151';b=D/'packed-regional-native-158'
parse=lambda p:[float(x) for x in re.findall(r'resting_native_profile step=\d+ command_ms=[\d.]+ gpu_ms=([\d.]+)',(p/'install-motion.stdout.log').read_text())]
ta,tb=parse(a),parse(b)
report={'baseline':a.name,'candidate':b.name,'baseline_accepted_steps':len(ta),'candidate_accepted_steps':len(tb),'step_s':1.5625e-5,'candidate_simulated_s':len(tb)*1.5625e-5,'accepted_gpu_ms':sum(tb),'gpu_only_real_time_factor':len(tb)*1.5625e-5/(sum(tb)/1000) if tb else None,'common_prefix_steps':min(len(ta),len(tb)),'common_prefix_gpu_ms':{'baseline':sum(ta[:len(tb)]),'candidate':sum(tb[:len(ta)])},'CSV_common_prefix':[],'claim':'Exact regional scheduling only. Exploratory matched prefix timing; no supported-rest, sustained motion, presentation or0.1x qualification. Full fine-FEM endpoint force/energy parity not measured.'}
for x in sorted((a/'install-motion').rglob('*.csv')):
 y=b/'install-motion'/x.relative_to(a/'install-motion')
 if not y.exists():continue
 with x.open() as f:aa=list(csv.reader(f))
 with y.open() as f:bb=list(csv.reader(f))
 n=min(len(aa),len(bb));same=aa[:n]==bb[:n]
 report['CSV_common_prefix'].append({'path':str(x.relative_to(a/'install-motion')),'baseline_rows':len(aa)-1,'candidate_rows':len(bb)-1,'common_rows':n-1,'common_rows_equal':same})
report['accepted_gpu_ms_uninstrumented_roots1plus']={'baseline':sum(ta[1:len(tb)]),'candidate':sum(tb[1:len(ta)])}
if (b/'install-motion.execution.json').exists():report['execution']=json.loads((b/'install-motion.execution.json').read_text())
if (b/'install-motion/resting-human.mrvpack').exists():
 report['MRV']=[]
 for rel in ('resting-human.mrvpack','accepted-geometry/step-0.mrvpack','accepted-geometry/step-32.mrvpack'):
  p0,p1=a/'install-motion'/rel,b/'install-motion'/rel
  if p0.exists() and p1.exists():
   h0=hashlib.file_digest(p0.open('rb'),'sha256').hexdigest();h1=hashlib.file_digest(p1.open('rb'),'sha256').hexdigest()
   report['MRV'].append({'path':rel,'baseline_sha256':h0,'candidate_sha256':h1,'byte_identical':h0==h1})
(D/'comparison-151-158.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('CSV_common_prefix','execution','MRV')},indent=2))
print('CSV bitwise',sum(x['common_rows_equal'] for x in report['CSV_common_prefix']),'/',len(report['CSV_common_prefix']))
if report.get('MRV'):print('MRV byte identical',all(x['byte_identical'] for x in report['MRV']))

