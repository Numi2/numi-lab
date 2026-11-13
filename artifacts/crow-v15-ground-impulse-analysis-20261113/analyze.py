from pathlib import Path
import gzip,json,collections,math
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts'); A=R/'crow-v15-ground-impulse-analysis-20261113'
inputs={'push':R/'crow-v15-stop-walk-toes-on-launch-20261113/push-nowings-20261127-rebound-no-walk-toes/run.stderr.gz'}
def load(path):
    total=collections.defaultdict(lambda:[0.,0.,0.,0.,0]); checks={}; state={}; materials=set()
    with gzip.open(path,'rt') as f:
        for line in f:
            if line.startswith('CROW_GROUND_CONTACT '):
                z=line.split(); t,k,e,a,b=map(int,z[1:6]);
                if not 348<=t<=375:continue
                v=list(map(float,z[8:])); assert len(v)==16,(len(v),z)
                n=v[4:7];u=v[7:10];mu_s,mu_d,jn,ju,jv=v[10:15];w=[n[1]*u[2]-n[2]*u[1],n[2]*u[0]-n[0]*u[2],n[0]*u[1]-n[1]*u[0]]
                assert a==35 or b==35
                sign=-1 if b==35 else 1
                impulse=[sign*(n[i]*jn+u[i]*ju+w[i]*jv) for i in range(3)]
                x=total[(e,t,k)];
                for i in range(3):x[i]+=impulse[i]
                x[3]+=math.hypot(ju,jv);x[4]+=1;materials.add((round(mu_s,6),round(mu_d,6)))
            elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
                z=line.split();t,k,e=map(int,z[1:4]);
                if 348<=t<=375:checks[(e,t,k)]=sum(map(float,z[6:8]))+float(z[10])
            elif line.startswith('CROW_SUPPORT_SAMPLE '):
                x=json.loads(line.split(' ',1)[1]);t=x['step'];e=x['lane']
                if t in (347,357,365,375):state[(e,t)]=x
    assert len(checks)==8*28*16,(path,len(checks));assert len(state)==8*4
    residual=[abs(total[key][2]-value) for key,value in checks.items()]; print(path.name,'maximum impulse reconstruction error',max(residual),'materials',materials)
    assert max(residual)<2e-7
    return total,state,materials,max(residual)
D={name:load(path) for name,path in inputs.items()}
rows=[]
for e in range(2,8):
    row={'lane':e}
    for name,(total,state,mat,err) in D.items():
        phases={}
        for label,lo,hi in [('early',348,357),('middle',358,365),('late',366,375),('whole',348,375)]:
            arr=[total[(e,t,k)] for t in range(lo,hi+1) for k in range(16)]
            phases[label]={'impulse_xyz_ns':[sum(v[i] for v in arr) for i in range(3)],'tangent_impulse_magnitudes_sum_ns':sum(v[3] for v in arr),'positive_contact_records':sum(v[4] for v in arr)}
        row[name]={'phases':phases,'root':{str(t):{'position':state[(e,t)]['q'][:3],'velocity':state[(e,t)]['v'][:3]} for t in (347,357,365,375)}}
    rows.append(row)
out={'scope':'Retrospective ground-contact decomposition, one development seed with correlated lanes; diagnostic, not a causal traction or qualification test.  Each vertical reconstruction independently checked against aggregate contact logger.  Root state is not whole-body COM.','validation':{n:{'material_pairs':list(map(list,d[2])),'max_vertical_reconstruction_error_ns':d[3]} for n,d in D.items()},'rows':rows}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:
 print('lane',r['lane'],'push impulses xyz early/middle/late',[[round(x,5) for x in r['push']['phases'][p]['impulse_xyz_ns']] for p in ('early','middle','late')],'root vz357/365/375',[round(r['push']['root'][str(t)]['velocity'][2],4) for t in (357,365,375)])
