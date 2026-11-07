#!/usr/bin/env python3
"""FP64 row-coupling references; not a device or full-world qualification."""
import unittest


def one_sweep(group_sizes, incoming, regularization=.01):
    # Each group is internally identical; distinct groups are orthogonal.
    count=sum(group_sizes)
    old_uniform=(1+regularization)/count
    old=[];coupled=[]
    for size in group_sizes:
        diagonal=1.
        row_sum=float(size)
        normal_preconditioned=incoming/(diagonal+regularization)
        local_alpha=min(1.,(1+regularization)*diagonal/row_sum)
        old.append(incoming-size*old_uniform*normal_preconditioned)
        coupled.append(incoming-size*local_alpha*normal_preconditioned)
    return old,coupled


class CouplingReferences(unittest.TestCase):
    def test_identical_and_orthogonal_groups(self):
        for sizes in ([257],[86,86,85]):
            old,new=one_sweep(sizes,.01)
            self.assertLess(max(map(abs,new)),1e-14)
            if len(sizes)>1:
                self.assertGreater(max(map(abs,old)),.006)

    def test_nonnegative_normal_block_row_bound(self):
        # D^-1 W has absolute row sum 1 and is similar to the symmetric
        # PSD matrix D^-1/2 W D^-1/2. Its eigenvalues are therefore in [0,1].
        vectors=[(1.,0.),(.7,.7),(0.,1.),(-.6,.8)]
        w=[[sum(a*b for a,b in zip(x,y)) for y in vectors] for x in vectors]
        d=[sum(abs(x) for x in row) for row in w]
        for i,row in enumerate(w):
            self.assertAlmostEqual(sum(abs(x/d[i]) for x in row),1)
        for x in ((1,0,0,0),(1,-2,3,-4),(.2,.3,-.1,1)):
            quadratic=sum(x[i]*w[i][j]*x[j]/(d[i]*d[j])**.5
                          for i in range(4) for j in range(4))
            self.assertGreaterEqual(quadratic,-1e-12)


if __name__=="__main__":
    unittest.main()
