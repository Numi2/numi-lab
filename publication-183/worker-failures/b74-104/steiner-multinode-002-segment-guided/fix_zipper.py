from pathlib import Path
p=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b74-104/steiner-multinode-002-segment-guided/segment_patch_trial.py')
s=p.read_text()
old="""oo,oa=angle_order(outer_xy);ii,ia=angle_order(inner_xy);oi=len(oo)-1;ij=len(ii)-1;adv_o=0;adv_i=0;ann=[]
while adv_o<len(oo) or adv_i<len(ii):
 to=oa[oo[adv_o]] if adv_o<len(oo) else np.inf
 ti=ia[ii[adv_i]] if adv_i<len(ii) else np.inf
 if to<ti:
  j=oo[adv_o];ann.append([('o',oo[oi]),('o',j),('i',ii[ij])]);oi=j;adv_o+=1
 else:
  j=ii[adv_i];ann.append([('o',oo[oi]),('i',j),('i',ii[ij])]);ij=j;adv_i+=1
"""
new="""oo,oa=angle_order(outer_xy);ii,ia=angle_order(inner_xy);cur_o=oo[-1];cur_i=ii[-1];adv_o=0;adv_i=0;ann=[]
while adv_o<len(oo) or adv_i<len(ii):
 to=oa[oo[adv_o]] if adv_o<len(oo) else np.inf
 ti=ia[ii[adv_i]] if adv_i<len(ii) else np.inf
 if to<ti:
  j=oo[adv_o];ann.append([('o',cur_o),('o',j),('i',cur_i)]);cur_o=j;adv_o+=1
 else:
  j=ii[adv_i];ann.append([('o',cur_o),('i',j),('i',cur_i)]);cur_i=j;adv_i+=1
"""
if old not in s: raise SystemExit('target zipper block not found')
p.write_text(s.replace(old,new))
print('patched annulus cursor to track vertex ids directly')

