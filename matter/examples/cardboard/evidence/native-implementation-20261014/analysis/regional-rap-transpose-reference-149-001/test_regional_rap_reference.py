#!/usr/bin/env python3
import math
import unittest
import regional_rap_reference as rr


def mm(a, b): return rr.matmul(a, b)
def eye(n): return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
def zeros(r,c): return [[0.0]*c for _ in range(r)]
def close_vec(a,b,tol=1e-10):
    return max((abs(x-y) for x,y in zip(a,b)), default=0.0) <= tol

def solve_dense_independent(A,b):
    # Small independent Gauss-Jordan oracle, separate from the captured LU solve.
    n=len(A); aug=[list(map(float,A[i]))+[float(b[i])] for i in range(n)]
    for k in range(n):
        p=max(range(k,n),key=lambda i:abs(aug[i][k]))
        if abs(aug[p][k])<1e-14: raise ValueError('singular oracle matrix')
        aug[k],aug[p]=aug[p],aug[k]
        d=aug[k][k]; aug[k]=[x/d for x in aug[k]]
        for i in range(n):
            if i==k: continue
            f=aug[i][k]
            aug[i]=[aug[i][j]-f*aug[k][j] for j in range(n+1)]
    return [aug[i][-1] for i in range(n)]

def independent_operator_oracle(P,AP,B,k):
    n=len(P); Pt=rr.transpose([row[:k] for row in P]); Q=[row[:k] for row in AP]
    R=mm(Pt,Q); columns=[]
    for j in range(n):
        e=[1.0 if i==j else 0.0 for i in range(n)]
        coarse_rhs=[Pt[r][j] for r in range(k)]
        c=solve_dense_independent(R,coarse_rhs)
        coarse=rr.matvec([row[:k] for row in P],c)
        aq=rr.matvec(Q,c)
        residual=[e[i]-aq[i] for i in range(n)]
        fine=rr.matvec(B,residual)
        columns.append([fine[i]+coarse[i] for i in range(n)])
    return rr.transpose(columns)

class RegionalRAPTransposeTests(unittest.TestCase):
    def setUp(self):
        self.n = 9
        self.identity = rr.Identity('source', 'program', 'world', 'candidate-state', 7, 12, 3, 0x3dcccccd)
        self.generation = rr.Generation(4, 2, 7, 12, 3, 11, 0x3dcccccd, 0)
        # Nonsymmetric A, with a first-column RAP pivot swap for base P=[e0,e1].
        self.A = [[0.0]*9 for _ in range(9)]
        for i in range(9): self.A[i][i] = 4.0 + .17*i
        self.A[0][0] = .05; self.A[0][1] = 2.0; self.A[1][0] = 1.0; self.A[1][1] = 3.0
        self.A[0][2] = -.2; self.A[1][2] = .3; self.A[2][0] = .1; self.A[2][1] = -.4
        self.A[3][0] = .12; self.A[4][1] = -.08; self.A[5][2] = .07
        self.Pbase = zeros(9,3)
        self.Pbase[0][0] = 1.0; self.Pbase[1][1] = 1.0
        # A nonzero padded slot verifies base modeCount is honored when stride=3.
        self.Pbase[2][2] = 9.0
        self.APbase = mm(self.A, self.Pbase)
        self.Paug = [row[:] for row in self.Pbase]
        self.restart_mode_E = [0.0,0.0,1.0,.1,-.2,.05,.03,0.0,.02]
        for i,v in enumerate(self.restart_mode_E): self.Paug[i][2] = v
        self.APaug = mm(self.A, self.Paug)
        self.Bblocks = [
            [[.50,.20,-.04],[-.10,.42,.30],[.08,-.20,.60]],
            [[.70,.10,.05],[-.05,.90,.15],[.20,.00,.80]],
            [[.60,-.15,.20],[.10,.55,-.08],[.03,.22,.75]],
        ]
        rows=[]
        for b in self.Bblocks:
            rows.append(tuple(tuple(list(r)+[1.0 if i==0 else 0.0]) for i,r in enumerate(b)))
        self.block_rows=tuple(rows)
        self.diag=tuple((.25,.25,.25) for _ in range(3))
        self.mask=((1,1,1),)*3
        self.base=self._variant('base',self.Pbase,self.APbase,2,3,self.generation.linearization_key)
        self.aug=self._variant('current-restart',self.Paug,self.APaug,3,3,self.generation.linearization_key,
            rr.Generation(4,2,7,12,3,11,0x3dcccccd,1),True)
        self.capture=rr.Capture(self.identity,self.base,self.aug,True,
            self.block_rows,self.diag,self.mask,True,True)

    def _variant(self,label,P,AP,k,stride,lin,gen=None,valid=None):
        Pt=rr.transpose([row[:k] for row in P])
        R=mm(Pt, [row[:k] for row in AP])
        return rr.Variant(label,P,AP,k,stride,rr.factor_rap(R),lin,gen,valid)

    def test_base_action_and_reverse_composition_match_dense_reference(self):
        c=rr.apply_regional(self.capture,[.2,-1.,.7,.3,-.8,.9,.5,.1,-.4],
            expected_identity=self.identity,expected_generation=self.generation)
        S=rr.dense_regional_operator(self.capture,expected_identity=self.identity,
            expected_generation=self.generation)
        B=rr.make_fine_inverse(self.capture,True)
        S_oracle=independent_operator_oracle(self.Pbase,self.APbase,B,2)
        self.assertTrue(all(close_vec(S[i],S_oracle[i]) for i in range(9)))
        x=[.3,-.4,.7,1.2,-.2,.5,-.8,.6,.9]
        self.assertTrue(close_vec(c,rr.matvec(S,[.2,-1.,.7,.3,-.8,.9,.5,.1,-.4])))
        ST=rr.transpose(S)
        yt=rr.apply_regional_transpose(self.capture,x,expected_identity=self.identity,
            expected_generation=self.generation)
        self.assertTrue(close_vec(yt,rr.matvec(ST,x)))
        self.assertAlmostEqual(rr.dot(c,x), rr.dot([.2,-1.,.7,.3,-.8,.9,.5,.1,-.4],yt), places=11)
        self.assertGreater(max(abs(S[i][j]-S[j][i]) for i in range(9) for j in range(9)),1e-3)
        lhs=rr.dot(rr.apply_regional(self.capture,x,expected_identity=self.identity,
            expected_generation=self.generation),[.2,-1.,.7,.3,-.8,.9,.5,.1,-.4])
        rhs=rr.dot(x,rr.apply_regional_transpose(self.capture,[.2,-1.,.7,.3,-.8,.9,.5,.1,-.4],
            expected_identity=self.identity,expected_generation=self.generation))
        self.assertAlmostEqual(lhs,rhs,places=11)
        self.assertEqual(self.base.factor.pivots,(1,0))

    def test_restart_augmentation_selected_only_for_current_restart_generation(self):
        g=rr.Generation(4,2,7,12,3,11,0x3dcccccd,1)
        selected=rr.select_variant(self.capture,self.identity,g)
        self.assertEqual(selected.label,'current-restart')
        x=[.2,-1.,.7,.3,-.8,.9,.5,.1,-.4]
        y=rr.apply_regional(self.capture,x,expected_identity=self.identity,expected_generation=g)
        S=rr.dense_regional_operator(self.capture,expected_identity=self.identity,expected_generation=g)
        self.assertTrue(close_vec(y,rr.matvec(S,x)))
        self.assertNotEqual(y,rr.apply_regional(self.capture,x,expected_identity=self.identity,
            expected_generation=self.generation))
        with self.assertRaisesRegex(rr.ReferenceError,'generation mismatch'):
            rr.select_variant(self.capture,self.identity,rr.Generation(4,2,7,12,3,11,0x3dcccccd,2))
        with self.assertRaisesRegex(rr.ReferenceError,'identity mismatch'):
            rr.select_variant(self.capture,rr.Identity('other','program','world','candidate-state',7,12,3,0x3dcccccd),g)

    def test_factor_transpose_solve_uses_final_row_permutation(self):
        k=2; P=[row[:k] for row in self.Pbase]; AP=[row[:k] for row in self.APbase]
        R=mm(rr.transpose(P),AP); f=rr.factor_rap(R)
        b=[.4,-1.2]
        x=rr.solve_rap_transpose(f,b)
        self.assertTrue(close_vec(rr.matvec(rr.transpose(R),x),b,1e-12))
        y=rr.solve_rap(f,b)
        self.assertTrue(close_vec(rr.matvec(R,y),b,1e-12))
        self.assertNotEqual(x,y)

    def test_base_restartzero_uses_modecount_not_padded_stride(self):
        y=rr.apply_regional(self.capture,[1.,.5,-.2,.1,.3,-.7,.8,-.1,.4],
            expected_identity=self.identity,expected_generation=self.generation)
        baseP=[row[:2] for row in self.Pbase]
        baseAP=[row[:2] for row in self.APbase]
        R=mm(rr.transpose(baseP),baseAP)
        f=rr.factor_rap(R)
        rhs=[math.fsum(baseP[i][j]*[1.,.5,-.2,.1,.3,-.7,.8,-.1,.4][i] for i in range(9)) for j in range(2)]
        c=rr.solve_rap(f,rhs)
        bc=rr.matvec(baseP,c); ac=rr.matvec(baseAP,c)
        B=rr.make_fine_inverse(self.capture,True)
        expected=[bc[i]+rr.matvec(B,[[1.,.5,-.2,.1,.3,-.7,.8,-.1,.4][j]-ac[j] for j in range(9)])[i] for i in range(9)]
        self.assertTrue(close_vec(y,expected))

    def test_invalid_augmentation_is_only_admitted_with_zero_extra_mode_words(self):
        p=[row[:] for row in self.Paug]; ap=[row[:] for row in self.APaug]
        for row in p: row[2]=0.0
        for row in ap: row[2]=0.0
        bad_aug=self._variant('invalid-current-restart',p,ap,3,3,
            self.generation.linearization_key,
            rr.Generation(4,2,7,12,3,11,0x3dcccccd,1),False)
        c=rr.Capture(self.identity,self.base,bad_aug,True,self.block_rows,self.diag,self.mask,True,True)
        g=rr.Generation(4,2,7,12,3,11,0x3dcccccd,1)
        rr.select_variant(c,self.identity,g)
        p[0][2]=1.0
        malformed=self._variant('invalid-but-nonzero',p,ap,3,3,
            self.generation.linearization_key,
            rr.Generation(4,2,7,12,3,11,0x3dcccccd,1),False)
        c2=rr.Capture(self.identity,self.base,malformed,True,self.block_rows,self.diag,self.mask,True,True)
        with self.assertRaisesRegex(rr.ReferenceError,'retained nonzero P/AP'):
            rr.select_variant(c2,self.identity,g)

    def test_unhealthy_or_unselected_is_zero_and_invalid_RAP_falls_back_to_B(self):
        x=[1.,-2.,.5,.3,.1,-.4,.2,.8,-.9]
        unhealthy=rr.Capture(self.identity,self.base,self.aug,True,self.block_rows,self.diag,self.mask,True,False)
        self.assertEqual(rr.apply_regional(unhealthy,x,expected_identity=self.identity,expected_generation=self.generation),[0.0]*9)
        unselected=rr.Capture(self.identity,self.base,self.aug,True,self.block_rows,self.diag,self.mask,False,True)
        with self.assertRaisesRegex(rr.ReferenceError,'not selected'):
            rr.apply_regional(unselected,x,expected_identity=self.identity,expected_generation=self.generation)
        invalid=rr.Variant(self.base.label,self.base.basis_storage,self.base.ap_storage,2,3,
            rr.Factor(self.base.factor.lu,self.base.factor.pivots,False),self.base.linearization)
        c=rr.Capture(self.identity,invalid,None,False,self.block_rows,self.diag,self.mask,True,True)
        b=rr.make_fine_inverse(c,True)
        self.assertTrue(close_vec(rr.apply_regional(c,x,expected_identity=self.identity,
            expected_generation=self.generation),rr.matvec(b,x)))
        self.assertTrue(close_vec(rr.apply_regional_transpose(c,x,expected_identity=self.identity,
            expected_generation=self.generation),rr.matvec(rr.transpose(b),x)))

    def test_block_marker_fallback_fixed_mask_and_bad_generation(self):
        rows=list(self.block_rows)
        rows[1]=tuple((0.,0.,0.,0.) for _ in range(3)) # source fallback to scalar D^-1
        diag=((.25,.25,.25),(.8,.7,.6),(.25,.25,.25))
        c=rr.Capture(self.identity,self.base,None,False,tuple(rows),diag,self.mask,True,True)
        x=[.2,.1,-.3,.8,-.4,.6,-.9,.5,.2]
        S=rr.dense_regional_operator(c,expected_identity=self.identity,expected_generation=self.generation)
        self.assertTrue(close_vec(rr.apply_regional_transpose(c,x,expected_identity=self.identity,
            expected_generation=self.generation),rr.matvec(rr.transpose(S),x)))
        Sdiag=rr.dense_regional_operator(c,expected_identity=self.identity,
            expected_generation=self.generation,use_block=False)
        ydiag=rr.apply_regional_transpose(c,x,expected_identity=self.identity,
            expected_generation=self.generation,use_block=False)
        self.assertTrue(close_vec(ydiag,rr.matvec(rr.transpose(Sdiag),x)))
        badmask=((1,1,1),(0,0,0),(1,1,1))
        fixedrows=list(self.block_rows); fixedrows[1]=tuple((0.,0.,0.,0.) for _ in range(3))
        fixedP=[r[:] for r in self.Pbase]; fixedAP=[r[:] for r in self.APbase]
        for i in range(3,6):
            fixedP[i]=[0.,0.,0.]; fixedAP[i]=[0.,0.,0.]
        # New base variant/factor for same captured operator under this test's fixed rows.
        v=self._variant('fixed',fixedP,fixedAP,2,3,self.generation.linearization_key)
        fc=rr.Capture(self.identity,v,None,False,tuple(fixedrows),((.25,.25,.25),(0.,0.,0.),(.25,.25,.25)),badmask,True,True)
        out=rr.apply_regional(fc,x,expected_identity=self.identity,expected_generation=self.generation)
        self.assertEqual(out[3:6],[0.0,0.0,0.0])
        badgen=rr.Capture(self.identity,self.base,None,False,self.block_rows,self.diag,self.mask,True,True)
        with self.assertRaisesRegex(rr.ReferenceError,'another control/substep/Newton'):
            rr.apply_regional(badgen,x,expected_identity=self.identity,
                expected_generation=rr.Generation(4,2,7,12,3,12,0x3dcccccd,0))

if __name__=='__main__':
    unittest.main(verbosity=2)
