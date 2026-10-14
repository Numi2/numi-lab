#!/usr/bin/env python3
"""Path-free Codex access to the same local NumiLab tissue workspace."""
import argparse,json,sys,urllib.request,urllib.error,urllib.parse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',nargs='?',default='open',choices=['list','status','context','catalog','select','propose','edit','undo','revise','seal','reveal','replay','cancel','recover','history','snapshot','export','restore','open','readout']);p.add_argument('--revision',type=int);p.add_argument('--full',action='store_true',help='Include full retained registration metadata');p.add_argument('--id');p.add_argument('--assay');p.add_argument('--specimen');p.add_argument('--population');p.add_argument('--condition');p.add_argument('--name');p.add_argument('--port',type=int,default=None);p.add_argument('--foreground',action='store_true',help='Keep the restored service attached to this command');p.add_argument('--genes',nargs='+');p.add_argument('--targets',nargs='+');p.add_argument('--title');p.add_argument('--models',nargs='+');p.add_argument('--conditions',nargs='+');p.add_argument('--authorized-reveal',action='store_true',help='Use only after explicit user authorization to open observations');p.add_argument('--kind',choices=['gene','interventions','features','populations','objective-coverage'],default='gene');p.add_argument('--gene');p.add_argument('--target');p.add_argument('--limit',type=int,default=64);p.add_argument('--offset',type=int,default=0);p.add_argument('--search');p.add_argument('--normalization',choices=['raw_counts','log1p_10000'],default='raw_counts');a=p.parse_args()
    if a.action=='restore':
        from wet_lab_snapshot import restore
        print(json.dumps(restore(a.id,a.name),indent=2));return 0
    if a.action in ('open','list','status'):
        from wet_lab_entry import open_laboratory,laboratories,readiness
        try:
            if a.action=='open':result=open_laboratory(a.name,a.port,a.foreground)
            elif a.action=='list':result={'laboratories':laboratories()}
            else:
                active,health=readiness();result={'status':'ready' if active else 'offline','health':health,'recovery':'numi wet-lab open'}
            print(json.dumps(result,indent=2));return 0
        except (OSError,ValueError) as error:print(str(error),file=sys.stderr);return 2
    try:active=json.loads(__import__('wet_lab_entry').CONNECTION.read_text())
    except FileNotFoundError:p.error('Open the installed workspace with numi wet-lab first')
    body=None
    if a.action not in ('context','catalog','history','export','readout'):
        if a.revision is None:p.error('--revision from the latest context is required; never auto-retry stale edits')
        body={'action':'selection' if a.action=='select' else a.action,'expectedRevision':a.revision,'actor':'codex'}
        for k in ('id','genes','targets','title'):
            if getattr(a,k) is not None:body[k]=getattr(a,k)
        if a.models is not None:body['modelIDs']=a.models
        if a.conditions is not None:body['conditionIDs']=a.conditions
        if a.action=='select':body['selection']={'assayID':a.assay,'specimenID':a.specimen,'populationID':a.population}
        if a.action=='select' and a.condition:body['selection']['conditionID']=a.condition
        if a.action=='select' and a.gene:body['selection']['gene']=a.gene
        if a.action=='select' and a.target:body['selection']['target']=a.target
        if a.action=='reveal':
            if not a.authorized_reveal:p.error('Reveal is an explicit authorized operation; ask the user if they have not requested it')
            body['authorizeReveal']=True
    if a.action=='export':body={'id':a.id}
    endpoint='/api/'+a.action if a.action in ('snapshot','export') else '/api/shared'
    if a.action=='catalog':
        endpoint='/api/catalog?assay='+urllib.parse.quote(a.assay) if a.assay else '/api/assays'
    if a.action=='readout':
        context_request=urllib.request.Request(active['url']+'/api/shared',headers={'X-Wet-Lab-Token':active['token']})
        with urllib.request.urlopen(context_request,timeout=120) as response:state=json.load(response)
        selection=state.get('selection')
        if not selection:p.error('Select a dataset population first')
        if a.assay and a.assay!=selection['assayID']:p.error('Select the requested assay first; readout preserves shared population context')
        query={'kind':a.kind,'mode':'measured','limit':a.limit,'offset':a.offset,'normalization':a.normalization}
        for key in ('gene','target','search'):
            if getattr(a,key) is not None:query[key]=getattr(a,key)
        body={'assayID':selection['assayID'],'selection':selection,'query':query};endpoint='/api/readout'
    request=urllib.request.Request(active['url']+endpoint,data=json.dumps(body).encode() if body else None,headers={'Content-Type':'application/json','X-Wet-Lab-Token':active['token']})
    try:
        with urllib.request.urlopen(request,timeout=1800 if a.action in ('snapshot','export') else 120) as r:result=json.load(r)
    except urllib.error.HTTPError as e:print(e.read().decode(),file=sys.stderr);return 2
    except OSError as e:print('Workspace unavailable. Start numi wet-lab. '+str(e),file=sys.stderr);return 2
    if not a.full and isinstance(result,dict) and 'drafts' in result:
        for card in result['drafts']:
            card['undoDepth']=len(card.pop('undo',[]))
            for axis in card.get('axisResults',[]):axis.get('result',{}).pop('registration',None)
            card.get('result',{}).pop('registration',None)
        if a.action!='history':result['history']=result.get('history',[])[-12:]
    print(json.dumps(result['history'] if a.action=='history' else result,indent=2));return 0

if __name__=='__main__':sys.exit(main())
