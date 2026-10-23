from pathlib import Path
import hashlib,json,re,struct
R=Path('/Users/n/cardboard-opencourant-20261023/adhesive-fix')
sha=lambda b:hashlib.sha256(b).hexdigest()
checks=[]
def normalize(b):
 n=struct.unpack('>I',b[:4])[0];o=8+n
 assert b[4+n:8+n]==b[:4] and struct.unpack('>I',b[o:o+4])[0]==80
 start=o+4; stamp=b[start:start+24]
 assert re.fullmatch(rb'[A-Za-z]{3} [A-Za-z]{3} [ 0-9][0-9] [0-9]{2}:[0-9]{2}:[0-9]{2} [0-9]{4}',stamp),stamp
 return b[:start]+bytes(24)+b[start+24:],stamp.decode(),start
for other,stage in [('valgrind-staged-birth-03-stage2',2),('valgrind-staged-birth-03-stage3',3),('staged-bond-split-child-01',3)]:
 a=next((R/'runs/staged-bond-birth-03').glob(f'*T{stage:02d}'));b=next((R/'runs'/other).glob(f'*T{stage:02d}'))
 aa,bb=a.read_bytes(),b.read_bytes();na,da,o=normalize(aa);nb,db,ob=normalize(bb)
 checks.append(dict(run=other,native_history=a.name,reference_sha256=sha(aa),compared_sha256=sha(bb),raw_bit_identical=aa==bb,reference_timestamp=da,compared_timestamp=db,timestamp_byte_offset=o,timestamp_bytes=24,all_other_bytes_identical=na==nb,normalized_sha256=sha(na)))
out={'schema':'cardboard.restart-diagnostic-identity.v2','rule':'Only the validated ASCII wall-clock timestamp in the second Fortran record is masked; every other byte including all histories remains compared. Original files retained.','checks':checks}
(R/'checkpoint-and-diagnostic-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
