#!/usr/bin/env python3
import unittest
from decimal import Decimal as D
import benchmark as b

class SourceEquationTests(unittest.TestCase):
    def test_literal_and_recentered_switch_limits(self):
        left=b.E6*(1/b.E7-1)
        self.assertEqual(b.w_comp(D(1),"paper_absolute"),left)
        self.assertEqual(b.w_tension(D(1)),D(0))
        self.assertEqual(b.w_comp(D(1),"recentered"),D(0))
        self.assertLess(left,D(0))
    def test_offset_changes_A_but_not_branch_stress_or_tangent(self):
        lam=D("0.99"); shift=b.E6*(1-1/b.E7)
        self.assertLess(abs((b.w_comp(lam,"recentered")-b.w_comp(lam,"paper_absolute"))-shift),D("1e-68"))
        for alpha in (D(0),D("0.25"),D("0.96")):
            h=D("1e-24")
            for convention in ("paper_absolute","recentered"):
                fd=(b.w_comp(lam+h,convention)-b.w_comp(lam-h,convention))/(2*h)
                self.assertLess(abs(fd-b.p_comp(lam,alpha)/(1-alpha)),D("1e-35"))
            self.assertLess(abs(b.p_comp(lam,alpha)-(1-alpha)*b.p_comp(lam)),D("1e-68"))
            self.assertLess(abs(b.tangent_comp(lam,alpha)-(1-alpha)*b.tangent_comp(lam)),D("1e-68"))
    def test_energy_and_stress_derivative(self):
        for lam in (D("0.8"),D("0.99"),D("0.999")):
            h=D("1e-24")
            fd=(b.w_comp(lam+h,"paper_absolute")-b.w_comp(lam-h,"paper_absolute"))/(2*h)
            self.assertLess(abs(fd-b.p_comp(lam)),D("1e-40"))
            h2=D("1e-22")
            fd2=(b.p_comp(lam+h2)-b.p_comp(lam-h2))/(2*h2)
            self.assertLess(abs(fd2-b.tangent_comp(lam)),D("1e-35"))
    def test_damage_conjugate_uses_absolute_psi0(self):
        lam=D("0.99"); alpha=D("0.25"); h=D("1e-30"); psi=b.w_comp(lam,"paper_absolute")
        fd=(((1-(alpha+h))*psi)-((1-(alpha-h))*psi))/(2*h)
        self.assertLess(abs(fd+psi),D("1e-39")); self.assertLess(psi,D(0))
    def test_switch_tangents_are_distinct_and_literal_energy_jumps(self):
        self.assertEqual(b.tangent_comp(D(1)),D("16.79"))
        self.assertEqual(b.tangent_tension(D(1)),D("56.8"))
        eps=D("1e-20")
        self.assertLess(b.w_comp(D(1)-eps,"paper_absolute"),D("-0.333260869565"))
        self.assertLess(abs(b.w_comp(D(1)-eps,"recentered")),D("1e-18"))
    def test_prior_builder_check_values_are_binary32_constant_replay(self):
        psi=b.w_comp_with_float32_graph_constants(D("0.99"),"paper_absolute")
        self.assertLess(abs(psi*1e6-(-332362.550556)),1e-6)
        self.assertLess(abs(psi*.75*1e6-(-249271.912917)),1e-6)
        exact=float(b.w_comp(D("0.99"),"paper_absolute"))*1e6
        self.assertGreater(abs(exact-psi*1e6),0.005)

if __name__=="__main__": unittest.main(verbosity=2)
