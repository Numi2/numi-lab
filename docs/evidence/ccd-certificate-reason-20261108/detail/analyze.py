import json,pathlib
p=pathlib.Path(__file__).parent
runs={x:json.loads((p/(x+'.json')).read_text())['runs'] for x in ('baseline','instrumented')}
def traces(name):
 out=[]; case=None
 for line in (p/(name+'.stderr')).read_text().splitlines():
  if line.startswith('ccd_case='):case=json.loads(line.split('=',1)[1])
  elif line.startswith('ccd_event_trace='):
   assert case is not None
   out.append((case,json.loads(line.split('=',1)[1])))
 return out
tr={x:traces(x) for x in runs}
errors=[];diagnostics=[]
if len(runs['baseline'])!=12 or len(runs['instrumented'])!=12:errors.append('wrong case count')
for a,b in zip(runs['baseline'],runs['instrumented']):
 a=dict(a);b=dict(b);a.pop('elapsed_with_initialization',None);b.pop('elapsed_with_initialization',None)
 if a!=b:errors.append({'case':[a.get('fixture'),a.get('scale')],'run_difference':{k:[a.get(k),b.get(k)] for k in set(a)|set(b) if a.get(k)!=b.get(k)}})
if len(tr['baseline'])!=len(tr['instrumented']):errors.append('trace length mismatch')
for (ca,a),(cb,b) in zip(tr['baseline'],tr['instrumented']):
 if ca!=cb:errors.append('trace case mismatch');continue
 a=dict(a);b=dict(b)
 if b['pair_flags']&128:
  enc,alpha=b['pair_time']; reason=int(enc)//65536; corner=(int(enc)%65536)//8192; bits=(int(enc)%8192)//64; pending=int(enc)%64
  diagnostics.append({'case':ca,'pass':b['pass'],'status':b['status'],'reason':{1:'budget',2:'stack',3:'unrepresentable_bisection'}.get(reason,'unknown'),'corner':corner,'last_node_positive':bool(bits&1),'negative_witness':None if bits&1 else bool(bits&2),'positive_time':None if bits&1 else bool(bits&4),'positive_left_lower':None if bits&1 else bool(bits&8),'left_upper_inside_band':None if bits&1 else bool(bits&16),'pending_nodes':pending,'normalized_last_node_lower':alpha,'reported_work':b['pair_work']})
  a.pop('pair_time');b.pop('pair_time')
 if a!=b:errors.append({'case':ca,'pass':a['pass'],'trace_difference':[k for k in set(a)|set(b) if a.get(k)!=b.get(k)]})
output={'schema':'numi.ccd.certificate-reason.v1','cases':len(runs['baseline']),'trace_records':len(tr['baseline']),'comparison_pass':not errors,'errors':errors,'unresolved_records':diagnostics}
(p/'analysis.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
