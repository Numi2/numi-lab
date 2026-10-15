#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).with_name('audit_chain.py')
s=importlib.util.spec_from_file_location('audit_chain_tested',p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)

def test_exact_join():
    a=np.asarray([1.0,2.0],dtype='<f4')
    assert m.array_cmp(a,a.copy())['equal']

def test_join_rejects_signed_zero_even_if_numeric_equal():
    a=np.asarray([-0.0],dtype='<f4'); b=np.asarray([0.0],dtype='<f4')
    r=m.array_cmp(a,b)
    assert not r['equal'] and r['word_mismatches']==1 and r['signed_zero_only_words']==1

def test_join_rejects_physical_difference():
    a=np.asarray([1.0],dtype='<f4'); b=np.asarray([np.nextafter(np.float32(1),np.float32(2))],dtype='<f4')
    assert not m.array_cmp(a,b)['equal']

if __name__=='__main__':
    for f in (test_exact_join,test_join_rejects_signed_zero_even_if_numeric_equal,test_join_rejects_physical_difference): f()
    print('PASS 3 strict lineage tests')
