from pathlib import Path
import secrets,random,json,hashlib
A=Path("artifacts/crow-v15-final-layout-reservation-20261110");out=A/"reserved-layouts.json"
if out.exists():raise SystemExit("reservation exists; never regenerate")
layouts=[]
for name,sign in [("K",1),("L",-1)]:
 seed=secrets.randbits(64);r=random.Random(seed);gate_x=r.uniform(1.33,1.52);center=r.uniform(-.06,.06);half=r.uniform(.67,.72);a_x=r.uniform(2.56,2.74);b_x=r.uniform(3.76,3.94);p_x=r.uniform(4.98,5.16);a_y=sign*r.uniform(.32,.41);b_y=-sign*r.uniform(.32,.41);p_y=r.uniform(-.13,.13);p_z=r.uniform(.75,.84)
 positions=[[gate_x,center+half,.8],[gate_x,center-half,.8],[a_x,a_y,.65],[b_x,b_y,.65],[p_x,p_y,p_z]]
 layouts.append({"name":name,"generator_seed":seed,"reset_seed":secrets.randbits(32),"positions_m":[[round(v,6) for v in row] for row in positions]})
raw=(json.dumps({"schema":"numi.crow-unused-layout-reservation.v1","generator":"python.Random seeded by independent secrets; ranges fixed before draw","layouts":layouts},indent=2)+"\n").encode();out.write_bytes(raw);(A/"reservation.sha256").write_text(hashlib.sha256(raw).hexdigest()+"  reserved-layouts.json\n");print("reserved two new layouts; SHA-256",hashlib.sha256(raw).hexdigest())
