import math, unittest
from proxy import JackdawGlideProxy

class ProxyTest(unittest.TestCase):
    def test_heldout_speed(self):
        self.assertAlmostEqual(JackdawGlideProxy().drag_newtons(9.7,steady_glide=True,premoult=True),0.2318714,places=5)
    def test_rejects_slow_flight_and_nonfinite_values(self):
        for u in (0.0,0.5,12.0,float("nan"),float("inf")):
            with self.assertRaises(ValueError): JackdawGlideProxy().drag_newtons(u,steady_glide=True,premoult=True)
    def test_rejects_wrong_conditions(self):
        for glide,moult in ((False,True),(True,False),(False,False)):
            with self.assertRaises(ValueError): JackdawGlideProxy().drag_newtons(9.7,steady_glide=glide,premoult=moult)
if __name__=="__main__": unittest.main()
