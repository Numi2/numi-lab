import csv,json,pathlib
p=pathlib.Path(__file__).resolve().parent
root=p.parents[2]
def retained(n):
 path=(root/'docs/evidence/contact-mesh-reference-20261107/n16-h1e-06.json' if n==16
       else root/'docs/evidence/contact-n32-reference-20261107/n32-h1e-06.json')
 return json.loads(path.read_text())['force_bins_N']
def trace(n):
 with (p/f'n{n}-h1us.csv').open() as f:return list(csv.DictReader(f))
out={'gate':0.0025,'meshes':{}}
rows={n:trace(n) for n in (16,32)}
for n in (16,32):
 old=retained(n);peak=max(old);rs=rows[n]
 pressure=max(abs(float(r['force_from_pressure'])-v) for r,v in zip(rs,old))/peak
 momentum=max(abs(float(r['force_from_momentum'])-v) for r,v in zip(rs,old))/peak
 out['meshes'][str(n)]={'reference_peak_force_N':peak,'pressure_difference_fraction':pressure,
   'momentum_difference_fraction':momentum,'min_gap_m':min(float(r['min_gap']) for r in rs),
   'accounting_gate_pass':pressure<=.0025 and momentum<=.0025}
peak=max(retained(32))
difference=max(abs(float(a['force_from_momentum'])-float(b['force_from_momentum'])) for a,b in zip(rows[16],rows[32]))/peak
out['n16_n32_momentum_bin_difference_fraction']=difference
out['spatial_force_gate_pass']=difference<=.0025
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
