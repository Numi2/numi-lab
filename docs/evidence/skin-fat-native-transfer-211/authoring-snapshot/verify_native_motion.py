#!/usr/bin/env python3
"""Audit up to eight real accepted native skin snapshots with their matching bone poses."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time
from accepted_motion_snapshot import export_skin,read_receipt,sha

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--receipts',type=Path,nargs='+',required=True)
    ap.add_argument('--receipt-sha256',nargs='+',required=True)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--baseline-sha256',required=True)
    ap.add_argument('--exporter',type=Path,required=True)
    ap.add_argument('--exporter-sha256',required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if not 1<=len(a.receipts)<=8 or len(a.receipts)!=len(a.receipt_sha256):
        raise ValueError('require one explicit hash per snapshot, at most eight')
    if sha(a.baseline)!=a.baseline_sha256 or sha(a.exporter)!=a.exporter_sha256:
        raise ValueError('baseline or native exporter hash mismatch')
    records=[read_receipt(p,h) for p,h in zip(a.receipts,a.receipt_sha256)]
    steps=[r['accepted_step'] for r in records];times=[r['accepted_time_s'] for r in records]
    if any(b<=a for a,b in zip(steps,steps[1:])) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('accepted snapshots must have strictly increasing steps and times')
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();results=[]
    for path,pin,r in zip(a.receipts,a.receipt_sha256,records):
        directory=a.out/('step-'+str(r['accepted_step']))
        candidate,_=export_skin(path,pin,a.exporter,directory)
        command=[sys.executable,str(Path(__file__).with_name('verify_candidate.py')),
                 '--candidate',str(candidate),'--baseline',str(a.baseline),
                 '--accepted-receipt',str(path),'--world-bank-directory',str(directory/'accepted-world-banks'),'--out',str(directory/'ordered-gates')]
        cp=subprocess.run(command,capture_output=True,text=True,timeout=600)
        (directory/'verify.log').write_text(cp.stdout+cp.stderr)
        result={'receipt_sha256':pin,'accepted_step':r['accepted_step'],
                'accepted_time_s':r['accepted_time_s'],'accepted_root':r['accepted_root_fingerprint'],
                'return_code':cp.returncode,'candidate_sha256':sha(candidate)}
        results.append(result)
        receipt={'status':'PASS_CAPTURED_MOTION_GEOMETRY_ONLY' if cp.returncode==0 else 'REJECTED_CAPTURED_GEOMETRY',
                 'snapshots':results,'baseline_sha256':a.baseline_sha256,'exporter_sha256':a.exporter_sha256,
                 'wall_seconds':time.monotonic()-start,'continuous_motion_qualification':False,
                 'scope':'These sampled accepted surfaces do not establish between-sample validity or physical mass/cell/rollback qualification.'}
        (a.out/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if cp.returncode:
            print(json.dumps(receipt));return 2
    print(json.dumps(receipt));return 0
if __name__=='__main__':
    sys.exit(main())
