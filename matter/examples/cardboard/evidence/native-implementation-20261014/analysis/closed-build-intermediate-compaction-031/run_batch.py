#!/usr/bin/env python3
import importlib.util,json,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('compact_trees',str(HERE/'compact_trees.py'))
mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
doc=json.loads((HERE/'preflight.json').read_text())
log=HERE/'receipts'/'batch-progress.jsonl'
completed=0; skipped=0
with open(log,'a',encoding='utf-8') as out:
 for row in doc['targets']:
  key=row['key'];target=Path(row['target'])
  if (HERE/'receipts'/f'{key}-removed.json').exists() and (target/mod.POINTER).exists():
   continue
  before=mod.df_record()
  if before['free_bytes'] >= 2*1024**3:
   rec={'key':key,'status':'stopped_at_2GiB','df':before};out.write(json.dumps(rec,sort_keys=True)+'\n');out.flush();os_fsync=__import__('os').fsync;os_fsync(out.fileno());break
  if before['free_bytes'] < mod.MIN_FREE:
   rec={'key':key,'status':'stopped_below_512MiB_floor','df':before};out.write(json.dumps(rec,sort_keys=True)+'\n');out.flush();__import__('os').fsync(out.fileno());break
  try:
   prep,prep_sha,data=mod.make_inventory_archive(key,target,row)
   rem,rem_sha=mod.remove_target(key,row)
   rec={'key':key,'status':'removed_after_restore_verification','archive_bytes':prep['archive_bytes'],
        'removed_allocated_bytes':rem['removed_allocated_bytes'],'df_before':before,'df_after':rem['df_after_remove'],
        'prepare_receipt_sha256':prep_sha,'removal_receipt_sha256':rem_sha,'elapsed_s':round(time.time()-before.get('time',time.time()),3)}
   completed+=1
  except Exception as e:
   msg=str(e)
   rec={'key':key,'status':'skipped_or_failed_preserved','reason':msg,'df':mod.df_record()}
   skipped+=1
   if 'insufficient per-tree archive/restore headroom' not in msg and ('open handles' not in msg) and ('active build/probe' not in msg) and ('tracked paths' not in msg) and ('target already has restore pointer' not in msg):
    rec['traceback']=traceback.format_exc(limit=4)
  out.write(json.dumps(rec,sort_keys=True)+'\n');out.flush();__import__('os').fsync(out.fileno())
  if (completed+skipped)%5==0:
   print(json.dumps({'progress':completed+skipped,'completed':completed,'skipped':skipped,'last':rec,'df':mod.df_record()},sort_keys=True),flush=True)
print(json.dumps({'final_completed':completed,'final_skipped':skipped,'df':mod.df_record()},sort_keys=True),flush=True)
