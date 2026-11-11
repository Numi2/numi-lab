from pathlib import Path
import importlib.util,json,math
p=Path("artifacts/crow-v15-whole-gait-audit-20261111/audit.py")
s=importlib.util.spec_from_file_location("audit",p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
A=Path("artifacts/crow-v15-corvid-stride-audit-20261111")

def rotate(q,v):
 x,y,z,w=q; u=(x,y,z);d=sum(u[i]*v[i] for i in range(3));c=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);s=w*w-sum(t*t for t in u)
 return [s*v[i]+2*d*u[i]+2*w*c[i] for i in range(3)]
def anchor(body,local):
 v=rotate(body[3:7],local);return [body[i]+v[i] for i in range(3)]
def angle(x,y,z,rootq):
 u=a.rotate_inv(rootq,[x[i]-y[i] for i in range(3)]);v=a.rotate_inv(rootq,[z[i]-y[i] for i in range(3)])
 # Anatomical side-view angles, projected into torso sagittal plane.
 u=(u[0],u[2]);v=(v[0],v[2]);d=math.hypot(*u)*math.hypot(*v)
 if d<1e-10:raise ValueError("degenerate segment projection")
 return math.degrees(math.acos(max(-1.,min(1.,(u[0]*v[0]+u[1]*v[1])/d))))
def calc(body,q,e,t,side):
 get=lambda x:body[e,t,"crow_"+side+"_"+x]
 hip=anchor(get("thigh"),[0,0,.0316249989]);knee=anchor(get("shank"),[0,0,.0316249989]);ankle=anchor(get("foot"),[-.0287500005,0,.0092000002]);inter=anchor(get("front_toe_2_distal"),[-.00899999961,0,0]);neck=anchor([*q[:3],*q[3:7]],[.138650745,0,.0850982144])
 # Parent and child anchors must agree before treating them as anatomical landmarks.
 checks=[(hip,anchor(get("hip_roll_link"),[0,0,0])),(knee,anchor(get("thigh"),[0,0,-.0316249989])),(ankle,anchor(get("shank"),[0,0,-.0316249989])),(inter,anchor(get("front_toe_2_proximal"),[.00899999961,0,0]))]
 if any(math.dist(x,y)>1e-6 for x,y in checks):raise ValueError("parent-child joint anchor disagreement")
 return [angle(neck,hip,knee,q[3:7]),angle(hip,knee,ankle,q[3:7]),angle(knee,ankle,inter,q[3:7])]
def bouts(flags):
 out=[];start=None
 for i,v in enumerate(flags+[False]):
  if v and start is None:start=i
  if not v and start is not None:out.append([a.LO+start,a.LO+i-1]);start=None
 return out

def main():
 # Orthogonal sagittal geometry, and rotation inverse checks.
 assert abs(angle([0,0,1],[0,0,0],[1,0,0],[0,0,0,1])-90)<1e-9
 assert abs(angle([0,0,1],[0,0,0],[0,0,-1],[0,0,0,1])-180)<1e-9
 try: angle([0,1,0],[0,0,0],[1,0,0],[0,0,0,1]);raise AssertionError("degenerate angle passed")
 except ValueError:pass
 results=[]
 for seed,path in a.BASE.items():
  sup,sub,body,com=a.load(path)
  old=json.loads(Path("artifacts/crow-v15-whole-gait-audit-20261111/audit.json").read_text())
  expected=next(r["baseline"] for r in old["rows"] if r["seed"]==seed)
  for e in range(8):
   for si,side in enumerate(("left","right")):
    angles=[];rear=[];middle=[];sole=[]
    for t in range(a.LO,a.HI+1):
     angles.append(calc(body,sup[e,t]["q"],e,t,side))
     # Observer records toe normal impulse per physical toe segment, indices 5..20.
     rear.append(sum(sub[e,t,k][5+8*si+6]+sub[e,t,k][5+8*si+7] for k in range(a.N))>1e-7)
     middle.append(sum(sub[e,t,k][5+8*si+2]+sub[e,t,k][5+8*si+3] for k in range(a.N))>1e-7)
     sole.append(sum(sub[e,t,k][2+si] for k in range(a.N))>1e-7)
    swings=expected[e]["support_qualified_swing_bouts"][si];sample=[]
    for lo,hi in swings:
     z=angles[lo-a.LO:hi-a.LO+1]
     sample.append({"clearance_bout_endpoints":[lo,hi],"samples":len(z),"internal_angle_at_first_deg":z[0],"internal_angle_at_last_deg":z[-1],"sampled_min_step":[lo+min(range(len(z)),key=lambda k:z[k][j]) for j in range(3)],"angle_span_deg":[max(v[j] for v in z)-min(v[j] for v in z) for j in range(3)]})
    results.append({"seed":seed,"lane":e,"side":side,"window_angle_span_deg":[max(v[j] for v in angles)-min(v[j] for v in angles) for j in range(3)],"rear_toe_loaded_intervals":sum(rear),"middle_toe_loaded_intervals":sum(middle),"whole_foot_loaded_intervals":sum(sole),"rear_toe_loaded_bouts":bouts(rear),"middle_toe_loaded_bouts":bouts(middle),"clearance_bout_measurements":sample})
 out={"source":"retained development baseline; see prior replay validation","angle_method":"sagittal projection in root frame: neck-hip-knee, hip-knee-ankle, knee-ankle-middle interphalangeal; geometry uses physical joint anchors; not a species-calibrated anatomical model","event_method":"toe segment normal impulse summed over each 20ms interval, threshold 1e-7 N s; within-interval order and contact continuity unknown","limitations":"cleared bouts are endpoint-resolved swing candidates, not proven stride events; minima can be edge-truncated and sampling can miss extrema; model toe topology differs from bird; related magpie data are not crow targets","rows":results}
 (A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
 for key in ("rear_toe_loaded_intervals","middle_toe_loaded_intervals","whole_foot_loaded_intervals"):
  print(key,min(x[key] for x in results),max(x[key] for x in results))
 for i,n in enumerate(("hip","knee","ankle")):
  print(n,"window_angle_span_deg",min(x["window_angle_span_deg"][i] for x in results),max(x["window_angle_span_deg"][i] for x in results))
 print("swing candidates",sum(len(x["clearance_bout_measurements"]) for x in results))
if __name__=="__main__":main()
