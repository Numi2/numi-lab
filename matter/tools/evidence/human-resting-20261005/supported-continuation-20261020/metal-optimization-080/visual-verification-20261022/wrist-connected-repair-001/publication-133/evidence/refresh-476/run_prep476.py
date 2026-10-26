#!/usr/bin/env python3
"""Run the matched 476 connected-continuation preparation stage after 413/guard refresh."""
from __future__ import annotations
import hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT=Path('/Users/numi/Work/numi-human-connected-surface-069')
def write_json(path,obj):
    with Path(path).open('w') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n')
def main():
    argv=json.loads((ROOT/'prep-argv.json').read_text())
    log=ROOT/'prep-attempt-002.log'
    start=time.monotonic()
    with log.open('xb') as stream:
        proc=subprocess.run(argv,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,check=False)
    result={'stage':'prep','command':argv,'log_path':str(log),'returncode':proc.returncode,'seconds':time.monotonic()-start}
    write_json(ROOT/'prep-attempt-002-execution.json',result)
    print(json.dumps({k:result[k] for k in ('stage','returncode','seconds','log_path')},sort_keys=True))
    return proc.returncode
if __name__=='__main__': raise SystemExit(main())
