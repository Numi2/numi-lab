import json,struct
f32=lambda x:struct.unpack('f',struct.pack('f',x))[0]
def predict():
 rows=[]
 for variant,mass,arm in [(4,.009,.001),(5,.017,.002)]:
  M=f32(mass)+f32(arm)
  for step in [.000425,.000675,.000925]:
   h=f32(step)
   for lane,t in enumerate([-.00004,-.00007,-.0025,.00004]):
    target=f32(t);cap=f32(.01);free=max(-cap,min(cap,300*target/(1+(h*8+h*h*300)/M)))
    for contact in [False,True]:
     law=max(-cap,min(cap,300*target));constrained=contact and target<0
     rows.append({'variant':variant,'lane':lane,'contact':contact,'h':h,'mass':M,'target':target,'split_force_N':free,'ideal_coupled_force_N':law if constrained else free,'ideal_coupled_v_m_per_s':0 if constrained else h*free/M,'ideal_coupled_contact_impulse_Ns':-h*law if constrained else 0,'split_relative_motor_law_error':abs(free-law)/abs(law) if constrained else 0})
 return {'model':'Implicit bounded motor impulse plus unilateral contact in one native primal solve; impulse bounds +-h*cap, target K*(target-q0)/(D+h*K), row compliance h/(D+h*K).','rows':rows,'motor_law_error_gate_N':1e-7,'momentum_residual_gate_Ns':2e-9,'primary_reduction_fraction':.99}
if __name__=='__main__':print(json.dumps(predict(),indent=2,sort_keys=True))
