from __future__ import annotations
import hashlib
import pathlib
import sys
import unittest
import mpmath as mp
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
SOURCE173=BASE/'snapshots/source-checkpoint-warm-breakdown-173/matter/tools/cardboard_bilinear_fp64_reference.py'
EXPECTED_SOURCE173_SHA='a910339573822bf32c7ea895c295a0a51b8b5a99d901b1e3ab1c788c094f6995'
sys.path.insert(0,str(HERE))
import compensated_reference_v1 as v1
import compensated_reference_v2 as v2

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
class PortableReferenceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if sha(SOURCE173)!=EXPECTED_SOURCE173_SHA: raise RuntimeError('source173 reference SHA mismatch')
  x=np.load(HERE/'fixture_tet750.npz')
  cls.tet=int(x['tet'][0]); cls.p=x['parameters_f32'].astype(np.float64); cls.old=x['accepted_history_f32'].astype(np.float64)
  cls.F=x['F_center_fp64']; cls.zroot=x['mp_root_state_fp64']; cls.zbranches=x['fb_branch_states_fp64']
  cls.base=v1.import_reference(SOURCE173); cls.ref=v2.Reference(cls.base)
  mp.mp.dps=100
 def test_explicit_source_required_and_source_hash(self):
  self.assertEqual(sha(SOURCE173),EXPECTED_SOURCE173_SHA)
  with self.assertRaises(ValueError): v1.import_reference(None)
 def test_mixed_fb_quotient_and_direct_values_match_mp(self):
  values,parts,meta=v2.fb_row(np.repeat(self.p,2,axis=0),np.repeat(self.old,2,axis=0),self.zbranches)
  self.assertEqual(meta['fb_quotient_branch'].tolist(),[True,False])
  dl,y,eps,root=parts[2],parts[3],parts[4],parts[5]
  for i in range(2):
   md=mp.mpf(float(dl.hi[i]))+mp.mpf(float(dl.lo[i])); my=mp.mpf(float(y.hi[i]))+mp.mpf(float(y.lo[i])); me=mp.mpf(float(eps.hi[i]))+mp.mpf(float(eps.lo[i]))
   expected=mp.sqrt(md*md+my*my+me*me)-md-my
   self.assertEqual(float(values[i]),float(expected))
 def test_residual_hill_flow_and_fb_against_independent_mp(self):
  p=self.p; old=self.old; z=self.zroot; F=self.F
  actual=v2.residual(self.base,p,F,old,z)
  state=[mp.mpf(float(x)) for x in z[0]]; hist=[mp.mpf(float(x)) for x in old[0]]
  pp=[mp.mpf(float(x)) for x in p[0]]
  a,b,c,d,e,v=state[6:12]; ff,gg,hh,ll,mm,nn=(pp[i] for i in range(11,17))
  q2=ff*(b-c)**2+gg*(c-a)**2+hh*(a-b)**2+2*ll*d*d+2*mm*e*e+2*nn*v*v+pp[25]**2
  q=mp.sqrt(q2)
  flow=[(gg*(a-c)+hh*(a-b))/q,(ff*(b-c)+hh*(b-a))/q,(ff*(c-b)+gg*(c-a))/q,ll*d/q,mm*e/q,nn*v/q]
  dl=state[12]-hist[12]
  for k in range(6): self.assertLess(abs(mp.mpf(float(actual[0,k]))-(state[k]-hist[k]-dl*flow[k])),mp.mpf('2e-14'))
  y=1+pp[17]*pp[19]*pp[21]/pp[10]*state[13]+pp[18]*pp[20]*pp[22]/pp[10]*state[14]-q
  eps=pp[26]; fb=mp.sqrt(dl*dl+y*y+eps*eps)-dl-y
  self.assertEqual(float(actual[0,12]),float(fb))
 def test_qflow_and_fb_jacobian_rows_against_mp(self):
  p=self.p; f=self.F; old=self.old; z=self.zroot
  jac=v2.jacobian(self.base,p,f,old,z)[0]
  mm=lambda x:mp.mpf(float(x))
  pp=p[0].astype(np.float64); qstate=z[0]
  ff,gg,hh,ll,mcoef,nn=(mm(pp[i]) for i in range(11,17))
  a,b,c,d,e,v=(mm(x) for x in qstate[6:12]); dl=mm(qstate[12])-mm(old[0,12]); eps=mm(pp[25])
  q2=ff*(b-c)**2+gg*(c-a)**2+hh*(a-b)**2+2*ll*d*d+2*mcoef*e*e+2*nn*v*v+eps*eps
  q=mp.sqrt(q2); numer=[gg*(a-c)+hh*(a-b),ff*(b-c)+hh*(b-a),ff*(c-b)+gg*(c-a),ll*d,mcoef*e,nn*v]
  dq2=[-2*gg*(c-a)+2*hh*(a-b),2*ff*(b-c)-2*hh*(a-b),-2*ff*(b-c)+2*gg*(c-a),4*ll*d,4*mcoef*e,4*nn*v]
  dnum=[[gg+hh,-hh,-gg,0,0,0],[-hh,ff+hh,-ff,0,0,0],[-gg,-ff,ff+gg,0,0,0],[0,0,ll,0,0,0],[0,0,0,0,mcoef,0],[0,0,0,0,0,nn]]
  # Source 15-state q-coordinate block is compared at captured MP root.
  expected=[[-dl*(dnum[i][j]/q-numer[i]*dq2[j]/(2*q**3)) for j in range(6)] for i in range(6)]
  for i in range(6):
   for j in range(6): self.assertLess(abs(mm(jac[6+i,6+j])-expected[i][j]),mp.mpf('2e-11'))
  self.assertTrue(np.isfinite(jac[12]).all())
  # The FB row's positive-dl/positive-y quotient derivative is independently stable.
  _,parts,_=v2.fb_row(p,old,z); dlp=mp.mpf(float(parts[2].hi[0]))+mp.mpf(float(parts[2].lo[0])); yp=mp.mpf(float(parts[3].hi[0]))+mp.mpf(float(parts[3].lo[0])); ep=mm(pp[26]); rt=mp.sqrt(dlp*dlp+yp*yp+ep*ep)
  self.assertLess(abs(mm(jac[12,12])-(dlp/rt-1)),mp.mpf('2e-14'))
if __name__=='__main__': unittest.main(verbosity=2)
