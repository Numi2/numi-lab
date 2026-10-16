import json, math, struct, hashlib
from decimal import Decimal, localcontext
from pathlib import Path
BASE=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
OUT=BASE/'native/contact-barrier-energy-replay-500/output.bin'
DD=BASE/'analysis/native500-compensated-feature-001/dd-output.jsonl'
assert hashlib.sha256(OUT.read_bytes()).hexdigest()=='cb1f014b70d5bd21e08f385e4fbea2c6d857328b2fd4a9c0506d64cb18ff0237'
rows=[json.loads(x) for x in DD.read_text().splitlines()]
def fword(buf,off): return struct.unpack_from('<f',buf,off)[0]
def d(x): return Decimal.from_float(float(x))
def dot(a,b): return sum((x*y for x,y in zip(a,b)),Decimal(0))
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def add(a,b): return tuple(x+y for x,y in zip(a,b))
def mul(s,a): return tuple(s*x for x in a)
def closest(p,a,b,c):
 ab,ac,ap=sub(b,a),sub(c,a),sub(p,a)
 d1,d2=dot(ab,ap),dot(ac,ap)
 if d1<=0 and d2<=0:return a,(Decimal(1),Decimal(0),Decimal(0)),'vertex_a'
 bp=sub(p,b);d3,d4=dot(ab,bp),dot(ac,bp)
 if d3>=0 and d4<=d3:return b,(Decimal(0),Decimal(1),Decimal(0)),'vertex_b'
 vc=d1*d4-d3*d2
 if vc<=0 and d1>=0 and d3<=0:
  v=d1/max(d1-d3,Decimal('1e-12'));return add(a,mul(v,ab)),(1-v,v,Decimal(0)),'edge_ab'
 cp=sub(p,c);d5,d6=dot(ab,cp),dot(ac,cp)
 if d6>=0 and d5<=d6:return c,(Decimal(0),Decimal(0),Decimal(1)),'vertex_c'
 vb=d5*d2-d1*d6
 if vb<=0 and d2>=0 and d6<=0:
  w=d2/max(d2-d6,Decimal('1e-12'));return add(a,mul(w,ac)),(1-w,Decimal(0),w),'edge_ac'
 va=d3*d6-d5*d4
 if va<=0 and d4-d3>=0 and d5-d6>=0:
  w=(d4-d3)/max((d4-d3)+(d5-d6),Decimal('1e-12'))
  return add(b,mul(w,sub(c,b))),(Decimal(0),1-w,w),'edge_bc'
 den=max(va+vb+vc,Decimal('1e-12'));v,w=vb/den,vc/den
 return add(add(a,mul(v,ab)),mul(w,ac)),(1-v-w,v,w),'face'
def vec(buf,offset):return tuple(d(fword(buf,offset+i*4)) for i in range(3))
with localcontext() as ctx:
 ctx.prec=90
 data=OUT.read_bytes()
 for row in range(8):
  rec=row*224
  for label,po,to,no in [('current',0,16,196),('contact_limiter',64,80,204),('fem_applied',128,144,212)]:
   p=vec(data,rec+po);a=vec(data,rec+to);b=vec(data,rec+to+16);c=vec(data,rec+to+32)
   q,bary,branch=closest(p,a,b,c);delta=sub(p,q);sep=dot(delta,delta).sqrt()
   entry=next(x for x in rows if x['row']==row and x['endpoint']==label)
   hi,lo=map(float,entry['distance_words'])
   dd=Decimal.from_float(hi)+Decimal.from_float(lo)
   print(json.dumps({'row':row,'endpoint':label,'decimal_branch':branch,'dd_branch':entry['branch'],
       'native':fword(data,rec+no),'decimal_sep':str(sep),'dd_sep':str(dd),
       'dd_abs_error':str(abs(dd-sep)),'native_abs_error':str(abs(d(fword(data,rec+no))-sep))},sort_keys=True))
