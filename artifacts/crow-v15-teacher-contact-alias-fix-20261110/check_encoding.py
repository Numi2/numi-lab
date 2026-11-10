from pathlib import Path
import json
# Every representable millimetre/s bin over the declared command domain.
max_error=0
for x in range(1001):
 for y in range(601):
  encoded=x+1024*y;decoded_x=encoded%1024;decoded_y=encoded//1024;max_error=max(max_error,abs(decoded_x-x),abs(decoded_y-y))
assert max_error==0
Path('artifacts/crow-v15-teacher-contact-alias-fix-20261110/encoding.json').write_text(json.dumps({'tested_bins':1001*601,'max_bin_error':max_error,'max_encoded':1000+1024*600,'qualification':'integer packing only; GPU command path and student observation isolation require separate tests'},indent=2)+'\n')
