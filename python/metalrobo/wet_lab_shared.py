"""Shared draft edits and local operations for the existing Wet Lab service.

Biological compilation, prediction, observation and replay stay in NumiVivo.
The revision is a compare-and-swap token, never a last-writer-wins timestamp.
"""
import copy,fcntl,json,os,signal,subprocess,sys,time,uuid
from contextlib import contextmanager
from pathlib import Path

class Conflict(ValueError): pass

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp');
    with os.fdopen(os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:
        json.dump(v,f,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

class SharedWorkspace:
    def __init__(self,root,config):
        self.root=Path(root);self.folder=self.root/'shared';self.folder.mkdir(exist_ok=True);self.config=config
        with self.lock():
            if not self.path.exists():write(self.path,{'revision':0,'selection':None,'drafts':[],'operations':[],'history':[]})
            write(self.folder/'service.json',config)
    @property
    def path(self):return self.folder/'workspace.json'
    @contextmanager
    def lock(self):
        with (self.folder/'workspace.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX);yield
    def state(self):
        self.reconcile()
        with self.lock():
            state=read(self.path)
            state['models']=self.models()
            return self.enrich(state)
    @staticmethod
    def selection(sel,targets=None,genes=None):
        return {'specimen':sel['specimenID'],'population':sel['populationID'],'condition':sel.get('conditionID','measured-endpoint'),'targets':targets or [],'objective':{'genes':genes or [sel.get('gene') or 'Clu'],'preserveGenes':[],'penalty':0}}
    def enrich(self,state):
        if not self.config.get('vivoRoot'):return state
        import laboratory
        from investigation import model_difference
        sel=state.get('selection')
        if sel and sel['assayID'] in self.config['assays']:
            state['selectionSupport']=laboratory.population_support(self.config['assays'][sel['assayID']],self.selection(sel),[sel.get('gene') or 'Clu'])
        for d in state['drafts']:
            d['decisionSupport']=laboratory.population_support(self.config['assays'][d['selection']['assayID']],self.selection(d['selection']),d['genes'])
            d['modelDifference']=model_difference(d)
        return state
    def models(self):
        if not self.config['assays']:return []
        import laboratory
        result=[]
        for identifier,path in self.config['assays'].items():
            a=read(path);cap=laboratory.capabilities(Path(path))
            if not cap['prediction']:continue
            result.append({'id':identifier,'version':a.get('modelVersion',identifier),'title':a['title'],
              'evidence':a.get('evidence','MODEL INFERENCE'),'biologicalPromotion':False,
              'presentation':cap['presentation'],'conditions':cap['conditions'],
              'specimens':[{k:v for k,v in x.items() if k not in ('source','sourceSHA256')} for x in cap['specimens']],
              'populations':cap['populations'],'featureCount':len(cap.get('features',[]))})
        return result
    def event(self,s,event):
        s['revision']+=1;s['history'].append({'time':time.time(),'revision':s['revision'],**event});write(self.path,s)
    def mutate(self,b):
        with self.lock():
            s=read(self.path)
            if b.get('expectedRevision')!=s['revision']:raise Conflict('Human or agent changed this workspace. Read context and review the edits before retrying.')
            action=b['action'];draft=None
            if action=='selection':
                sel=b['selection'];config=self.config['assays'].get(sel['assayID'])
                if not config:raise ValueError('Unknown assay')
                import laboratory
                cat=laboratory.catalog(Path(config))
                if sel['specimenID'] not in {v['id'] for v in cat.get('specimens',[])} or sel['populationID'] not in {v['id'] for v in cat.get('populations',[])}:raise ValueError('Unknown specimen or population')
                conditions={c['id'] for c in cat['conditions']}
                if sel.get('conditionID','measured-endpoint') not in conditions:raise ValueError('Unsupported condition')
                pop=next(v for v in cat['populations'] if v['id']==sel['populationID'])
                if pop.get('specimenID',sel['specimenID'])!=sel['specimenID'] or pop.get('conditionID',sel.get('conditionID','measured-endpoint'))!=sel.get('conditionID','measured-endpoint'):raise ValueError('Population does not match specimen and condition')
                s['selection']=sel
            elif action=='propose':
                if not s['selection']:raise ValueError('Select a biological population first')
                draft={'id':uuid.uuid4().hex,'status':'draft','selection':copy.deepcopy(s['selection']),'genes':b['genes'],'targets':b.get('targets',[]),'title':b.get('title','Reduce selected RNA program'),'undo':[],'parent':b.get('parent'),'modelIDs':b.get('modelIDs',[s['selection']['assayID']]),'conditionIDs':b.get('conditionIDs',[s['selection'].get('conditionID','measured-endpoint')])}
                self.qualify(draft);s['drafts'].append(draft)
            else:
                draft=next((d for d in s['drafts'] if d['id']==b.get('id')),None)
                if not draft:raise ValueError('Unknown draft')
                if action in ('edit','undo'):
                    if draft['status']!='draft':raise ValueError('Sealed/running cards are immutable. Create a revision.')
                    if action=='undo':
                        if not draft['undo']:raise ValueError('Nothing to undo')
                        prior=draft['undo'].pop();draft.update(prior)
                    else:
                        draft['undo'].append({k:copy.deepcopy(draft.get(k,[draft['selection']['assayID']] if k=='modelIDs' else ['measured-endpoint'] if k=='conditionIDs' else None)) for k in ('genes','targets','title','modelIDs','conditionIDs')})
                        for k in ('genes','targets','title','modelIDs','conditionIDs'):
                            if k in b:draft[k]=b[k]
                    self.qualify(draft)
                elif action=='revise':
                    new=copy.deepcopy(draft);new.update(id=uuid.uuid4().hex,status='draft',parent=draft['id'],undo=[])
                    for k in ('record','operation','result','axisResults'):new.pop(k,None)
                    self.qualify(new);s['drafts'].append(new);draft=new
                elif action in ('seal','reveal','replay','recover'):
                    if action=='recover':
                        op=next(o for o in s['operations'] if o['id']==draft['operation'])
                        if op['status'] not in ('failed','cancelled','interrupted'):raise ValueError('Operation has not stopped')
                        if self.alive(op):raise ValueError('Original operation is still running')
                        action=op['action']
                    if action=='seal':
                        if draft['status'] not in ('draft','interrupted'):raise ValueError('Already sealed or running')
                        before=copy.deepcopy(draft.get('axes'))
                        self.qualify(draft)
                        if before is not None and [a.get('binding') for a in before]!=[a.get('binding') for a in draft['axes']]:raise Conflict('Model artifact changed after proposal. Review a new model version before sealing.')
                        if not draft['coverage']['canExecute'] or not draft['targets']:raise ValueError('Unsupported objective or empty comparison')
                    elif not draft.get('record'):raise ValueError('Seal first')
                    if action=='reveal' and b.get('authorizeReveal') is not True:raise ValueError('Observation reveal requires explicit user authorization; candidate inspection cannot reveal')
                    if any(o['status']=='running' for o in s['operations']):raise ValueError('A local experiment is running; cancel or await it first')
                    op={'id':uuid.uuid4().hex,'action':action,'draftID':draft['id'],'status':'running','startedAt':time.time(),'progress':'Native owner '+action+' started; observations stay closed unless reveal was authorized'}
                    job=self.folder/(op['id']+'.json');write(job,{'config':self.config,'draft':draft,'operation':op})
                    with (self.folder/(op['id']+'.log')).open('w') as log:
                        process=subprocess.Popen([sys.executable,__file__,'worker',str(self.root),str(job)],stdout=log,stderr=log,start_new_session=True)
                    op['pid']=process.pid;s['operations'].append(op);draft['operation']=op['id'];draft['status']='running'
                elif action=='cancel':
                    op=next(o for o in s['operations'] if o['id']==draft['operation'])
                    if op['status']!='running':raise ValueError('Operation is not running')
                    if self.alive(op):os.killpg(op['pid'],signal.SIGTERM)
                    op['status']='cancelled';op['progress']='Cancelled; partial artifacts retained. Recovery starts a new operation after process exit.';draft['status']='interrupted' if not draft.get('record') else 'sealed'
                else:raise ValueError('Unsupported workspace action')
            self.event(s,{'action':action,'draftID':draft['id'] if draft else None,'actor':b.get('actor','human')});s['models']=self.models();return self.enrich(s)
    def qualify(self,d):
        if not isinstance(d['genes'],list) or not 1<=len(d['genes'])<=64 or any(not isinstance(g,str) or not g.strip() or g!=g.strip() for g in d['genes']):raise ValueError('Objective must contain 1 to 64 explicit gene symbols')
        import laboratory
        from wetlab import sha
        sel=d['selection'];config=Path(self.config['assays'][sel['assayID']]);cat=laboratory.catalog(config)
        if not isinstance(d['targets'],list) or len(d['targets'])!=len(set(d['targets'])):raise ValueError('Distinct intervention list required')
        d['supportedTargets']=sorted({x['target'] for x in laboratory.population_support(config,self.selection(sel),d['genes'])['candidates'] if x['canExecute']})
        modelIDs=d.get('modelIDs',[sel['assayID']]);conditionIDs=d.get('conditionIDs',[sel.get('conditionID','measured-endpoint')])
        if not isinstance(modelIDs,list) or not 1<=len(modelIDs)<=4 or len(set(modelIDs))!=len(modelIDs):raise ValueError('Select one to four distinct model versions')
        if not isinstance(conditionIDs,list) or not 1<=len(conditionIDs)<=4 or len(set(conditionIDs))!=len(conditionIDs):raise ValueError('Select one to four distinct conditions')
        d['axes']=[]
        for mid in modelIDs:
            for condition in conditionIDs:
                axis={'modelID':mid,'conditionID':condition,'canExecute':False,'corrections':[]};path=self.config['assays'].get(mid)
                if not path:axis.update(reason='Model not registered',corrections=[{'modelID':m['id']} for m in self.models()])
                else:
                    complete=self.selection({**sel,'conditionID':condition},d['targets'],d['genes']);cap=laboratory.capabilities(Path(path))
                    try:
                        coverage=laboratory.inspect_objective(Path(path),complete,complete['objective']);axis['coverage']=coverage
                        laboratory.validate_selection(Path(path),complete)
                        artifact=read(path);reference=read(config);specimen=next(x for x in artifact['specimens'] if x['id']==sel['specimenID']);original=next(x for x in reference['specimens'] if x['id']==sel['specimenID'])
                        if specimen['sourceSHA256']!=original['sourceSHA256']:raise ValueError('Specimen source differs between model versions')
                        axis.update(canExecute=coverage['canExecute'],reason='Supported experimental inference',selection=complete,populationSupport=laboratory.population_support(Path(path),complete,d['genes']),binding={'assaySHA256':sha(path),'modelVersion':artifact.get('modelVersion',mid),'runtimeSHA256':artifact['runtime']['sha256'],'weights':{k:v['sha256'] for k,v in artifact['models'].items()},'featureAxisSHA256':sha(Path(path).parent/'features.json'),'specimenSHA256':specimen['sourceSHA256'],'populationID':sel['populationID'],'conditionID':condition,'scoringOwnerSHA256':sha(laboratory.scoring_owner(path)),'evidence':'MODEL INFERENCE','biologicalPromotion':False})
                    except (ValueError,KeyError,StopIteration) as error:
                        axis['reason']=str(error) or 'Specimen or population incompatible'
                        if condition not in {c['id'] for c in cap['conditions']}:axis['corrections']=[{'conditionID':c['id']} for c in cap['conditions']]
                        elif any(not g['eligible'] for g in axis.get('coverage',{}).get('genes',[])):axis['corrections']=[{'removeUnsupportedGenes':[g['gene'] for g in axis['coverage']['genes'] if not g['eligible']],'requiresObjectiveEdit':True}]
                        else:
                            support=laboratory.population_support(Path(path),complete,d['genes']);unsupported=[t for t in d['targets'] if not any(c['target']==t and c['role']=='direct' and c['canExecute'] for c in support['candidates'])]
                            axis['corrections']=[{'removeUnsupportedTargets':unsupported,'requiresCandidateEdit':True}] if unsupported else [{'reviewSupportedPopulations':cap['populations'],'reviewSupportedTargets':cap['targets'],'retainIntent':True}]
                d['axes'].append(axis)
        try:d['coverage']=laboratory.inspect_objective(config,self.selection(sel,d['targets'],d['genes']),{'genes':d['genes'],'preserveGenes':[],'penalty':0})
        except (ValueError,KeyError,StopIteration):d['coverage']={'genes':[],'candidateEffects':[]}
        d['coverage']['canExecute']=all(x['canExecute'] for x in d['axes'])
        d['coverage']['executionReason']='; '.join(x['reason'] for x in d['axes'] if not x['canExecute'])
    def alive(self,op):
        # Avoid signalling a reused PID; verify the exact local job argument.
        r=subprocess.run(['ps','-p',str(op.get('pid',0)),'-o','command='],capture_output=True,text=True)
        return str(self.folder/(op['id']+'.json')) in r.stdout
    def reconcile(self):
        with self.lock():
            s=read(self.path)
            for op in s['operations']:
                if op['status']=='running' and not self.alive(op):
                    op['status']='interrupted';op['progress']='Worker exited before recording completion; artifacts retained'
                    next(d for d in s['drafts'] if d['id']==op['draftID'])['status']='interrupted'
                    self.event(s,{'action':'interrupted','draftID':op['draftID']})

def worker(root,job):
    j=read(job);config=j['config'];d=j['draft'];op=j['operation'];sys.path.insert(0,config['vivoRoot']+'/Tools/VirtualWetLab');import laboratory
    owner=laboratory.adapter_for_config(Path(config['assays'][d['selection']['assayID']]));runtime={'binary':Path(config['binary']) if config.get('binary') else None};store=SharedWorkspace(root,config)
    try:
        axes=d.get('axes') or [{'modelID':d['selection']['assayID'],'conditionID':'measured-endpoint'}]
        axisResults=[]
        for axis in axes:
            mid=axis['modelID'];condition=axis['conditionID']
            with store.lock():
                state=read(store.path);operation=next(x for x in state['operations'] if x['id']==op['id'])
                if operation['status']!='running':raise ValueError('Operation no longer active')
                operation['progress']=op['action']+' '+mid+' / '+condition+' ('+str(len(axisResults)+1)+'/'+str(len(axes))+')'
                store.event(state,{'action':'progress','draftID':d['id']})
            from wetlab import sha
            assay=Path(config['assays'][mid])
            if axis.get('binding') and sha(assay)!=axis['binding']['assaySHA256']:raise ValueError('Registered model changed; create and review a new draft')
            owner=laboratory.adapter_for_config(assay)
            prior=next((r for r in d.get('axisResults',[]) if r['modelID']==mid and r['conditionID']==condition),None)
            verification=None
            sel={**d['selection'],'assayID':mid,'conditionID':condition};run=Path(root)/(prior['record'] if prior else d['record']) if prior or d.get('record') else None
            if op['action']=='seal':
                complete=store.selection(sel,d['targets'],d['genes'])
                # The preregistration is written and fsynced before native execution.
                prereg=store.folder/(op['id']+'-'+str(len(axisResults))+'-preregistration.json')
                registration={'format':'numilab-objective-preregistration/v2','genes':d['genes'],'selection':sel,'validatedSelection':complete,'comparisonAxes':axes,'modelBinding':axis.get('binding'),'coverage':d['coverage'],'draftID':d['id'],'objectiveOwnerSHA256':sha(laboratory.scoring_owner(assay))}
                write(prereg,registration);complete['preregistration']={'sha256':sha(prereg),'name':prereg.name}
                run=laboratory.predict(assay,runtime,Path(root),complete)
                write(run/'objective-registration.json',registration)
                write(run/'objective-seal.json',{'objectiveSHA256':sha(run/'objective-registration.json'),'predictionSealSHA256':sha(run/'seal.json')})
            elif op['action']=='reveal':
                from wetlab import sha
                if sha(run/'objective-registration.json')!=read(run/'objective-seal.json')['objectiveSHA256']:raise ValueError('Objective registration changed')
                if not (run/'comparison.json').exists():owner.reveal(run,runtime)
            elif op['action']=='replay':verification=owner.verify(run,runtime)
            if (run/'objective-registration.json').exists():
                from wetlab import sha
                objective=read(run/'objective-registration.json')
                if sha(run/'objective-registration.json')!=read(run/'objective-seal.json')['objectiveSHA256']:raise ValueError('Objective seal changed')
                if objective.get('objectiveOwnerSHA256') and objective['objectiveOwnerSHA256']!=sha(laboratory.scoring_owner(assay)):raise ValueError('Objective owner changed; use retained owner or create a new record')
            result=owner.summary(run)
            if verification is not None:result['verification']=verification
            result['objectiveEvaluation']=laboratory.evaluate_objective(assay,run,{'genes':d['genes'],'preserveGenes':[],'penalty':0})
            objective_path=run/('objective-evaluation.json' if result['revealed'] else 'objective-prediction.json')
            if objective_path.exists():
                if read(objective_path)!=result['objectiveEvaluation']:raise ValueError('Objective replay changed')
            else:write(objective_path,result['objectiveEvaluation'])
            axisResults.append({'modelID':mid,'conditionID':condition,'record':run.name,'binding':axis.get('binding'),'result':result})
        with store.lock():
            s=read(store.path);current=next(x for x in s['operations'] if x['id']==op['id']);card=next(x for x in s['drafts'] if x['id']==d['id'])
            if current['status']=='running':
                current.update(status='completed',progress=op['action']+' completed',finishedAt=time.time());card.update(status='revealed' if all(x['result']['revealed'] for x in axisResults) else 'sealed',record=axisResults[0]['record'],result=axisResults[0]['result'],axisResults=axisResults);store.event(s,{'action':op['action']+'-completed','draftID':d['id']})
    except Exception as error:
        with store.lock():
            s=read(store.path);current=next(x for x in s['operations'] if x['id']==op['id']);card=next(x for x in s['drafts'] if x['id']==d['id'])
            if current['status']=='running':current.update(status='failed',progress=str(error));card['status']='interrupted';store.event(s,{'action':'failed','draftID':d['id'],'error':str(error)})
        raise

if __name__=='__main__':worker(Path(sys.argv[2]),Path(sys.argv[3]))
