import ast
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix,csc_matrix
p=Path(__file__).with_name('rebind_joint64_mass_support.py')
tree=ast.parse(p.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_ancestry_receiver_maps')
exec(compile(ast.Module(body=[function],type_ignores=[]),str(p),'exec'))
rng=np.random.default_rng(320)
for case in range(32):
 nold=13;nnew=17;old=rng.normal(size=(nold,3));new=rng.normal(size=(nnew,3))
 if case%3==0:old[:3]=0;new[:3]=0
 entries=[np.unique(np.r_[rng.integers(nold,size=4),i%nold]) for i in range(nnew)]
 # Cover every parent; ties use exact old/new IDs.
 for i in range(nold):entries[i]=np.unique(np.r_[entries[i],i])
 off=np.r_[0,np.cumsum([len(x) for x in entries])];rr=np.concatenate(entries)
 nr=np.repeat(np.arange(nnew),np.diff(off));d=np.einsum('ij,ij->i',new[nr]-old[rr],new[nr]-old[rr])
 order=np.lexsort((rr,d,nr));hit=np.r_[True,np.diff(nr[order])!=0];expected=rr[order[hit]]
 order=np.lexsort((nr,d,rr));hit=np.r_[True,np.diff(rr[order])!=0];nearest=np.empty(nold,dtype=np.int64);nearest[rr[order[hit]]]=nr[order[hit]]
 missing=np.flatnonzero(np.bincount(expected,minlength=nold)==0)
 ref=csr_matrix((np.ones(nnew+len(missing),np.uint8),(np.r_[expected,missing],np.r_[np.arange(nnew),nearest[missing]])),shape=(nold,nnew))
 for chunk in [1,5,37,2000000]:
  chosen,mapping,full=build_ancestry_receiver_maps([(off,rr,0,0)],new,old,np.zeros(nnew),np.zeros(nold),chunk)
  assert np.array_equal(chosen,expected)
  assert (mapping!=ref).nnz==0
  truth=csr_matrix((np.ones(len(rr),np.uint8),(rr,nr)),shape=(nold,nnew))
  assert (full!=truth).nnz==0
print('PASS_128_BATCHED_ANCESTRY_COMPARISONS_INCLUDING_TIES_AND_TINY_CHUNKS')

# Distinct material partitions exercise both old/new offsets and global row joins.
old=np.array([[0,0,0],[1,0,0],[2,0,0],[0,1,0],[1,1,0]],float)
new=np.array([[.2,0,0],[.8,0,0],[1.8,0,0],[.2,1,0],[.8,1,0],[.9,1,0]],float)
unions=[(np.array([0,2,4,6]),np.array([0,1,0,1,1,2]),0,0),(np.array([0,2,4,6]),np.array([0,1,0,1,0,1]),3,3)]
for chunk in [1,3,100]:
 chosen,mapping,full=build_ancestry_receiver_maps(unions,new,old,np.array([15]*3+[16]*3),np.array([15]*3+[16]*2),chunk)
 assert np.array_equal(chosen,[0,1,2,3,4,4])
 assert full.shape==(5,6) and full.nnz==12
print('PASS_MULTIMATERIAL_GLOBAL_OFFSETS_AND_OWNERSHIP')
