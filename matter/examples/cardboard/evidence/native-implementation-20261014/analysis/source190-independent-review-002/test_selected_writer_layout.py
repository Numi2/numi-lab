#!/usr/bin/env python3
import hashlib
import pathlib
import unittest

SOURCE_PATH = pathlib.Path("/Users/n/cardboard-implementation-20261014-host-macmini/snapshots/source-contact-damped-cgls-memory-193-001/matter/src/metal/base_cgls_runtime.metalinc")
SOURCE_SHA256 = "1b9d6541fcd01bea803a70f878ba1bda462d05cfabefa0d1ebe3038c9aab66d9"

def supported_layout(E, N, U, active=None, fixed_nodes=()):
    if E <= 0 or N <= 0 or U != 2 * E * N:
        raise ValueError("unsupported CGLS row layout")
    if active is None:
        active=[True]*E
    if len(active)!=E:
        raise ValueError("environment activity shape mismatch")
    count=E*N
    writes=[[] for _ in range(E)]
    selected=[None]*E
    masks=[[0,0,0,0] for _ in range(U)]
    fixed=set(fixed_nodes)
    for g in range(U):
        env=(g % count)//N
        if g < count:
            free=(g not in fixed)
            if active[env] and free:
                masks[g]=[1,1,1,0]
            if g % N == 0:
                writes[env].append(g)
                selected[env]=1 if active[env] else 0
    return writes,selected,masks

class SelectedWriterLayoutTests(unittest.TestCase):
    def test_source_hash_and_mechanical_guard(self):
        data=SOURCE_PATH.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),SOURCE_SHA256)
        lines=data.decode().splitlines()
        stores=[i for i,line in enumerate(lines) if "selected[environment] =" in line]
        self.assertEqual(len(stores),1)
        store=stores[0]
        self.assertEqual(stores[0]+1,59)
        guard=max(i for i in range(store) if lines[i].strip()=="if (global < count) {")
        depth=0
        end=None
        for i in range(guard,len(lines)):
            depth+=lines[i].count("{")-lines[i].count("}")
            if i>guard and depth==0:
                end=i
                break
        self.assertIsNotNone(end)
        self.assertLess(store,end)
        self.assertTrue(any("selected[environment] !=" in line for line in lines[store+1:]))

    def test_one_mechanical_writer_per_environment(self):
        for E in (1,2,4):
            for N in (1,3,7):
                with self.subTest(E=E,N=N):
                    writes,selected,_=supported_layout(E,N,2*E*N)
                    self.assertEqual(writes,[[env*N] for env in range(E)])
                    self.assertEqual(selected,[1]*E)
                    self.assertTrue(all(0<=row<E*N for group in writes for row in group))

    def test_field_half_zero_mask_and_no_selected_writers(self):
        for E in (1,2,4):
            N=5
            writes,_,masks=supported_layout(E,N,2*E*N)
            self.assertTrue(all(mask==[0,0,0,0] for mask in masks[E*N:]))
            self.assertEqual(sum(len(w) for w in writes),E)
            self.assertTrue(all(row<E*N for group in writes for row in group))

    def test_fixed_first_node_does_not_gate_or_duplicate_selection(self):
        E,N=2,4
        writes,selected,masks=supported_layout(E,N,2*E*N,active=[True,True],fixed_nodes={0,4})
        self.assertEqual(writes,[[0],[4]])
        self.assertEqual(selected,[1,1])
        self.assertEqual(masks[0],[0,0,0,0])
        self.assertEqual(masks[4],[0,0,0,0])
        self.assertEqual(masks[1],[1,1,1,0])
        self.assertEqual(masks[5],[1,1,1,0])

    def test_inactive_environment_has_one_zero_value_writer(self):
        writes,selected,_=supported_layout(4,3,24,active=[True,False,True,False])
        self.assertEqual(writes,[[0],[3],[6],[9]])
        self.assertEqual(selected,[1,0,1,0])

    def test_invalid_shapes_are_rejected_by_cpu_contract(self):
        for E,N,U in ((0,3,0),(2,0,0),(2,3,11),(2,3,13),(1,1,1)):
            with self.subTest(E=E,N=N,U=U):
                with self.assertRaises(ValueError):
                    supported_layout(E,N,U)

if __name__=="__main__":
    unittest.main(verbosity=2)
