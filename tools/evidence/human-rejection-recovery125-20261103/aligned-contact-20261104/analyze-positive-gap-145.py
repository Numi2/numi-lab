from pathlib import Path
import csv,json,re,math,hashlib
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
a=D/'early-tangent-native-139';b=D/'positive-gap-native-145'
parse=lambda p:[float(x) for x in re.findall(r'resting_native_profile step=\d+ command_ms=[\d.]+ gpu_ms=([\d.]+)',(p/'install-motion.stdout.log').read_text())]
ta,tb=parse(a),parse(b)
report={'baseline':a.name,'candidate':b.name,'baseline_accepted_steps':len(ta),'candidate_accepted_steps':len(tb),'step_s':1.5625e-5,'candidate_simulated_s':len(tb)*1.5625e-5,'accepted_gpu_ms':sum(tb),'gpu_only_real_time_factor':len(tb)*1.5625e-5/(sum(tb)/1000),'common_prefix_steps':min(len(ta),len(tb)),'common_prefix_gpu_ms':{'baseline':sum(ta[:len(tb)]),'candidate':sum(tb[:len(ta)])},'CSV_common_prefix':[],'claim':'one extra accepted step; no sustained, resting or 0.1x qualification; numerical contact continuation changed; engineering checks incomplete without full fine-FEM endpoint comparison'}
for x in sorted((a/'install-motion').rglob('*.csv')):
 y=b/'install-motion'/x.relative_to(a/'install-motion')
 if not y.exists():continue
 with x.open() as f:aa=list(csv.reader(f))
 with y.open() as f:bb=list(csv.reader(f))
 n=min(len(aa),len(bb));same=aa[:n]==bb[:n]
 rr={'path':str(x.relative_to(a/'install-motion')),'baseline_rows':len(aa)-1,'candidate_rows':len(bb)-1,'common_rows':n-1,'common_rows_equal':same}
 if x.name=='resting-coupled.csv':
  cols=[]
  for j,h in enumerate(aa[0]):
   if j>=39:break
   dif=[];base=[]
   for u,v in zip(aa[1:n],bb[1:n]):
    try:f0,f1=float(u[j]),float(v[j])
    except (ValueError,IndexError):continue
    if math.isfinite(f0) and math.isfinite(f1):dif.append((f1-f0)**2);base.append(f0*f0)
   cols.append({'column':h,'L2_relative_difference':math.sqrt(sum(dif)/sum(base)) if sum(base)>0 else None,'L2_absolute_difference':math.sqrt(sum(dif))})
  rr['physiology']=cols
 report['CSV_common_prefix'].append(rr)
(D/'comparison-139-145.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='CSV_common_prefix'},indent=2))
print('CSV equal',sum(x['common_rows_equal'] for x in report['CSV_common_prefix']),'/',len(report['CSV_common_prefix']))

