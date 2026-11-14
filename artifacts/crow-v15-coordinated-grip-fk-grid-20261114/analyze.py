from pathlib import Path
import collections,json,math
A=Path('artifacts/crow-v15-coordinated-grip-fk-grid-20261114');data=collections.defaultdict(list)
for line in (A/'output.tsv').read_text().splitlines():
 e,t,*z=line.split();dh,dk,da,foot,front,rear=map(float,z);assert all(map(math.isfinite,(dh,dk,da,foot,front,rear)));data[int(e),int(t)].append({'offset_rad':[dh,dk,da],'gaps_m':[foot,front,rear]})
ref=json.loads((A/'pose-crosscheck.json').read_text())['placement'];ref={x['zero_based_tick']:[x['groups'][k]['min_box_gap_m'] for k in ('foot','front','rear')] for x in ref['samples']};rows=[]
for (e,t),xs in sorted(data.items()):
 zero=[x for x in xs if x['offset_rad']==[0.,0.,0.]];assert len(zero)==1 and t in ref
 err=max(abs(a-b) for a,b in zip(zero[0]['gaps_m'],ref[t]));cand=[x for x in xs if x['gaps_m'][0]>=-.0002 and all(-.001<=v<=.0002 for v in x['gaps_m'][1:])]
 cand.sort(key=lambda x:sum(y*y for y in x['offset_rad']));rows.append({'lane':e,'tick':t,'grid_points':len(xs),'baseline_gaps_m':zero[0]['gaps_m'],'baseline_max_abs_error_m':err,'qualifying_geometry_count':len(cand),'smallest_displacement_examples':cand[:5]})
out={'rows':rows,'all_zero_offsets_match_pose':len(rows)==3 and all(r['baseline_max_abs_error_m']<1e-6 for r in rows),'scope':'fixed-state nearest-collider geometry only; no dynamic feasibility, force or grip'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
