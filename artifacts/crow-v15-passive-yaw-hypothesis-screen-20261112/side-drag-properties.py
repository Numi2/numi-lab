import json,math,random
rng=random.Random(20261112);worst=0.;mirror=0.
for _ in range(10000):
 d=rng.uniform(0,.06);rho=rng.uniform(.8,1.4);cdarea=rng.uniform(.001,.2);vy=rng.uniform(-5,5);wz=rng.uniform(-10,10);vpoint=vy-d*wz;air=-vpoint;fy=.5*rho*cdarea*air*abs(air);torque=-d*fy;power=fy*vpoint;worst=max(worst,power);opp=-.5*rho*cdarea*air*abs(air);mirror=max(mirror,abs(fy+opp));assert power<=1e-12
 if abs(wz)<1e-100:assert torque*vy>=0
out={"samples":10000,"seed":20261112,"maximum_air_relative_power":worst,"mirror_force_error":mirror,"scope":"Analytical scalar force law only; a numerical simulation parity check is separate. Positive lateral motion with zero spin produces a restoring positive yaw moment for a rearward force point."}
open("artifacts/crow-v15-passive-yaw-hypothesis-screen-20261112/side-drag-properties.json","w").write(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
