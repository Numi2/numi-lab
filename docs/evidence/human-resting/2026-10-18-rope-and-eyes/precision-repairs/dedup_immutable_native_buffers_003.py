import os,json,hashlib,subprocess,time
from pathlib import Path
from collections import defaultdict
p=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-suspension-reference-probe-20261018-001").resolve()
reportpath=p/"verified-immutable-native-buffer-dedup-003.json";assert not reportpath.exists()
def available():return int(subprocess.check_output(["df","-kP",str(p)],text=True).splitlines()[-1].split()[3])*1024
def digest(path):
 with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
files=sorted(set(p.glob("attempt-*/native-run/*.bin"))|set(p.glob("*/common-cardiac-*.bin"))|set(p.glob("*/composed-receipt/common-cardiac-*.bin")))
groups=defaultdict(list)
for f in files:
 assert f.is_file() and not f.is_symlink();groups[f.stat().st_size].append(f)
before=available();retired=[];logical=0;allocated=0
for size,group in sorted(groups.items()):
 if len(group)<2:continue
 seen={}
 for f in group:
  st=f.stat();key=digest(f)
  if key not in seen:seen[key]=f;continue
  keep=seen[key];ks=keep.stat()
  if (st.st_dev,st.st_ino)==(ks.st_dev,ks.st_ino):continue
  assert st.st_dev==ks.st_dev and digest(keep)==key
  tmp=f.with_name(f.name+f".dedup-{os.getpid()}")
  assert not tmp.exists();os.link(keep,tmp);assert digest(tmp)==key and digest(f)==key;os.replace(tmp,f);assert digest(f)==key
  retired.append({"path":str(f),"canonical_path":str(keep),"sha256":key,"bytes":size,"retired_allocated_bytes":st.st_blocks*512,"prior_mtime_ns":st.st_mtime_ns})
  logical+=size;allocated+=st.st_blocks*512
after=available();report={"policy":"immutable generated native buffers only; atomic exact-SHA-verified hardlinks, every file path and byte retained; no sole inputs or captures retired","files_considered":len(files),"files_deduplicated":len(retired),"logical_duplicate_bytes_replaced":logical,"allocated_duplicate_bytes_retired":allocated,"df_available_before_bytes":before,"df_available_after_bytes":after,"df_available_delta_bytes":after-before,"apfs_note":"df is live container availability, and other system activity can affect the delta; allocated bytes do not independently prove APFS recovery","entries":retired}
reportpath.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({k:v for k,v in report.items() if k!="entries"},indent=2))
