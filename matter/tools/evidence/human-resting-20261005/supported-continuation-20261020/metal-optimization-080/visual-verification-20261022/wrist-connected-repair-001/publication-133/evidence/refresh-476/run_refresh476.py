#!/usr/bin/env python3
"""Run the pinned 476 depth, fixed-guard, and connected-input refresh sequentially."""
from __future__ import annotations
import hashlib, json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT=Path('/Users/numi/Work/numi-human-connected-surface-069')
def write_json(path, obj):
    with Path(path).open('w') as f:
        json.dump(obj,f,indent=2,sort_keys=True); f.write('\n')
def main():
    stages=(('depth','depth-argv.json'),('guards','guards-argv.json'),('prep','prep-argv.json'))
    results=[]
    for name, argv_file in stages:
        argv=json.loads((ROOT/argv_file).read_text())
        log=ROOT/(name+'.log')
        started=time.monotonic()
        with log.open('wb') as stream:
            proc=subprocess.run(argv,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,check=False)
        result={'stage':name,'command':argv,'log_path':str(log),'returncode':proc.returncode,'seconds':time.monotonic()-started}
        results.append(result)
        write_json(ROOT/(name+'-execution.json'),result)
        print(json.dumps({k:result[k] for k in ('stage','returncode','seconds','log_path')},sort_keys=True),flush=True)
        if proc.returncode:
            write_json(ROOT/'execution.json',{'status':'FAILED','stages':results})
            return proc.returncode
    write_json(ROOT/'execution.json',{'status':'PASS','stages':results})
    return 0
if __name__=='__main__':
    raise SystemExit(main())
