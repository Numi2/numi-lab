#!/usr/bin/env python3
"""Path-free Codex access to the same local NumiLab tissue workspace."""
import argparse,json,sys,urllib.request,urllib.error
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['context','select','propose','edit','undo','revise','seal','reveal','replay','cancel','recover','history']);p.add_argument('--revision',type=int);p.add_argument('--id');p.add_argument('--assay');p.add_argument('--specimen');p.add_argument('--population');p.add_argument('--genes',nargs='+');p.add_argument('--targets',nargs='+');p.add_argument('--title');p.add_argument('--authorized-reveal',action='store_true',help='Use only after explicit user authorization to open observations');a=p.parse_args()
    try:active=json.loads((Path.home()/'.numi/wet-lab-active.json').read_text())
    except FileNotFoundError:p.error('Open the installed workspace with numi wet-lab first')
    body=None
    if a.action not in ('context','history'):
        if a.revision is None:p.error('--revision from the latest context is required; never auto-retry stale edits')
        body={'action':'selection' if a.action=='select' else a.action,'expectedRevision':a.revision,'actor':'codex'}
        for k in ('id','genes','targets','title'):
            if getattr(a,k) is not None:body[k]=getattr(a,k)
        if a.action=='select':body['selection']={'assayID':a.assay,'specimenID':a.specimen,'populationID':a.population}
        if a.action=='reveal':
            if not a.authorized_reveal:p.error('Reveal is an explicit authorized operation; ask the user if they have not requested it')
            body['authorizeReveal']=True
    request=urllib.request.Request(active['url']+'/api/shared',data=json.dumps(body).encode() if body else None,headers={'Content-Type':'application/json','X-Wet-Lab-Token':active['token']})
    try:
        with urllib.request.urlopen(request,timeout=120) as r:result=json.load(r)
    except urllib.error.HTTPError as e:print(e.read().decode(),file=sys.stderr);return 2
    except OSError as e:print('Workspace unavailable. Start numi wet-lab. '+str(e),file=sys.stderr);return 2
    print(json.dumps(result['history'] if a.action=='history' else result,indent=2));return 0

if __name__=='__main__':sys.exit(main())
