#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
source=Path(sys.argv[1]); raw=source.read_bytes()
if not raw.endswith(bytes([92,110])): raise SystemExit('source suffix differs from reviewed malformed literal')
try: json.loads(raw)
except json.JSONDecodeError: pass
else: raise SystemExit('source unexpectedly parses; refusing correction')
doc=json.loads(raw[:-2])
encoded=(json.dumps(doc,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
json.loads(encoded)
print(json.dumps({'source_sha256':hashlib.sha256(raw).hexdigest(),'source_bytes':len(raw),'removed_suffix_hex':raw[-2:].hex(),'strict_json':True,'semantic_object_unchanged':json.loads(encoded)==doc,'corrected_sha256':hashlib.sha256(encoded).hexdigest(),'corrected_bytes':len(encoded)},sort_keys=True,allow_nan=False))
