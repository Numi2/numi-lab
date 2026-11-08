import hashlib,json,math,pathlib
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
a=root/'docs/evidence/contact-p2-n8-order18-20261108/fine.json';b=root/'docs/evidence/contact-p2-n16-restart-diskguard-20261108/fine.json';audit=root/'docs/evidence/contact-p2-n16-restart-diskguard-20261108/audit-result.json';verdict=root/'docs/evidence/contact-p2-n16-restart-diskguard-20261108/analysis.json'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
A=json.loads(a.read_text());B=json.loads(b.read_text());U=json.loads(audit.read_text());V=json.loads(verdict.read_text())
assert U['audit_pass'] and U['inputs']['reference_sha256']==sha(a) and U['phases']['fine']['output_sha256']==sha(b)
assert V['numerical_prerequisite_pass'] and not V['n8_to_n16_spatial']['pass']
assert A['n']==8 and B['n']==16 and A['variant']==B['variant']=='fine' and len(A['force_bins_N'])==len(B['force_bins_N'])==10
names=['time_s','force18_N','force48_N','force64_N','gap18_m','gap48_m','gap64_m','minJ','momentum_Ns','impulse_Ns','contact_work_J','contact_energy_J','total_energy_J']
assert A['trace_columns']==names and B['trace_columns'][:13]==names
rows=lambda x:{round(row[0]*1e7):dict(zip(x['trace_columns'],row)) for row in x['trace']}
x,y=rows(A),rows(B);keys=sorted(y);assert len(keys)==56 and keys[0]==45000 and keys[-1]==100000 and all(j-i==1000 for i,j in zip(keys,keys[1:])) and all(k in x for k in keys)
peak=max(B['force_bins_N']);bins=[];cumA=cumB=0.
for i,(u,v) in enumerate(zip(A['force_bins_N'],B['force_bins_N'])):
 cumA+=u*.001;cumB+=v*.001;bins.append({'bin_start_s':i*.001,'n8_force_N':u,'n16_force_N':v,'difference_N':v-u,'difference_over_n16_peak':(v-u)/peak,'n8_cumulative_impulse_Ns':cumA,'n16_cumulative_impulse_Ns':cumB})
assert math.isclose(cumA,A['impulse_Ns'],rel_tol=1e-10,abs_tol=1e-15) and math.isclose(cumB,B['impulse_Ns'],rel_tol=1e-10,abs_tol=1e-15)
def sampled(rows):
 active=[k for k in keys if rows[k]['force18_N']>0];episodes=[]
 for k in active:
  if not episodes or k-episodes[-1][-1]!=1000:episodes.append([])
  episodes[-1].append(k)
 return {'first_active_s':active[0]/1e7 if active else None,'active_at_final_sample':bool(active and active[-1]==keys[-1]),'active_episodes':[{'first_sample_s':e[0]/1e7,'last_sample_s':e[-1]/1e7} for e in episodes], 'min_gap64_m':min(rows[k]['gap64_m'] for k in keys),'min_J':min(rows[k]['minJ'] for k in keys),'final_contact_work_J':rows[keys[-1]]['contact_work_J'],'final_contact_energy_J':rows[keys[-1]]['contact_energy_J']}
maximum=max(keys,key=lambda k:abs(y[k]['force18_N']-x[k]['force18_N']))
z={'schema':'numi.contact.p2.fixed-law.mesh-episode.v1','source_hashes':{'n8_fine':sha(a),'n16_fine':sha(b),'audit':sha(audit),'spatial_analysis':sha(verdict)},'sample_interval_s':.0001,'n16_peak_force_bin_N':peak,'force_bins':bins,'n8':sampled(x),'n16':sampled(y),'maximum_sampled_force_difference':{'time_s':maximum/1e7,'n8_N':x[maximum]['force18_N'],'n16_N':y[maximum]['force18_N'],'difference_N':y[maximum]['force18_N']-x[maximum]['force18_N']},'registered_spatial_pass':False,'limitations':['Descriptive sampled trace comparison only; no exact onset, release or unsampled clearance guarantee.','The fixed regularization is not a physical material calibration.','No causal attribution of the spatial discrepancy follows.']}
(p/'analysis.json').write_text(json.dumps(z,indent=2,allow_nan=False)+'\n');print(json.dumps(z,indent=2,allow_nan=False))
