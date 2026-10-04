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
        with self.lock():return read(self.path)
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
                cat=laboratory.adapter_for_config(Path(config)).catalog(Path(config))
                if sel['specimenID'] not in {v['id'] for v in cat.get('specimens',[])} or sel['populationID'] not in {v['id'] for v in cat.get('populations',[])}:raise ValueError('Unknown specimen or population')
                s['selection']=sel
            elif action=='propose':
                if not s['selection']:raise ValueError('Select a tissue population first')
                draft={'id':uuid.uuid4().hex,'status':'draft','selection':copy.deepcopy(s['selection']),'genes':b['genes'],'targets':b.get('targets',[]),'title':b.get('title','Reduce selected RNA program'),'undo':[],'parent':b.get('parent')}
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
                        draft['undo'].append({k:copy.deepcopy(draft[k]) for k in ('genes','targets','title')})
                        for k in ('genes','targets','title'):
                            if k in b:draft[k]=b[k]
                    self.qualify(draft)
                elif action=='revise':
                    new=copy.deepcopy(draft);new.update(id=uuid.uuid4().hex,status='draft',parent=draft['id'],undo=[])
                    for k in ('record','operation','result'):new.pop(k,None)
                    self.qualify(new);s['drafts'].append(new);draft=new
                elif action in ('seal','reveal','replay','recover'):
                    if action=='recover':
                        op=next(o for o in s['operations'] if o['id']==draft['operation'])
                        if op['status'] not in ('failed','cancelled','interrupted'):raise ValueError('Operation has not stopped')
                        if self.alive(op):raise ValueError('Original operation is still running')
                        action=op['action']
                    if action=='seal':
                        if draft['status'] not in ('draft','interrupted'):raise ValueError('Already sealed or running')
                        self.qualify(draft)
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
            self.event(s,{'action':action,'draftID':draft['id'] if draft else None,'actor':b.get('actor','human')});return s
    def qualify(self,d):
        if not isinstance(d['genes'],list) or not 1<=len(d['genes'])<=64 or any(not isinstance(g,str) or not g.strip() or g!=g.strip() for g in d['genes']):raise ValueError('Objective must contain 1 to 64 explicit gene symbols')
        from objective_inspection import inspect
        import laboratory
        sel=d['selection'];config=Path(self.config['assays'][sel['assayID']]);cat=laboratory.adapter_for_config(config).catalog(config);targets={x['target'] for x in cat['targets']}
        if not isinstance(d['targets'],list) or len(d['targets'])!=len(set(d['targets'])) or not set(d['targets'])<=targets:raise ValueError('Unsupported or repeated intervention')
        d['supportedTargets']=sorted(targets);d['coverage']=inspect(config,{'specimen':sel['specimenID'],'population':sel['populationID']},d['genes'])
        specimen=next(x for x in cat['specimens'] if x['id']==sel['specimenID'])
        if not specimen.get('inferenceSupported',False):d['coverage'].update(canExecute=False,executionReason='Owner declares this specimen inspectable only; prediction unavailable')
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
        verification=None
        sel=d['selection'];run=Path(root)/d['record'] if d.get('record') else None
        if op['action']=='seal':
            run=owner.predict(Path(config['assays'][sel['assayID']]),runtime,Path(root),{'specimen':sel['specimenID'],'population':sel['populationID'],'targets':d['targets']})
            # Bind the exact molecular objective before any reveal. Existing owner seal stays immutable.
            from wetlab import sha
            import objective_inspection
            write(run/'objective-registration.json',{'genes':d['genes'],'selection':sel,'coverage':d['coverage'],'draftID':d['id'],'objectiveOwnerSHA256':sha(objective_inspection.__file__),'predictionSealSHA256':sha(run/'seal.json')})
            write(run/'objective-seal.json',{'objectiveSHA256':sha(run/'objective-registration.json')})
        elif op['action']=='reveal':
            from wetlab import sha
            if sha(run/'objective-registration.json')!=read(run/'objective-seal.json')['objectiveSHA256']:raise ValueError('Objective registration changed')
            if not (run/'comparison.json').exists():owner.reveal(run,runtime)
        elif op['action']=='replay':verification=owner.verify(run,runtime)
        if (run/'objective-registration.json').exists():
            from wetlab import sha
            import objective_inspection
            objective=read(run/'objective-registration.json')
            if sha(run/'objective-registration.json')!=read(run/'objective-seal.json')['objectiveSHA256']:raise ValueError('Objective seal changed')
            if objective.get('objectiveOwnerSHA256') and objective['objectiveOwnerSHA256']!=sha(objective_inspection.__file__):raise ValueError('Objective owner changed; use retained owner or create a new record')
        result=owner.summary(run)
        if verification is not None:result['verification']=verification
        from objective_inspection import evaluate
        result['objectiveEvaluation']=evaluate(Path(config['assays'][d['selection']['assayID']]),run,d['genes'])
        objective_path=run/('objective-evaluation.json' if result['revealed'] else 'objective-prediction.json')
        if objective_path.exists():
            if read(objective_path)!=result['objectiveEvaluation']:raise ValueError('Objective replay changed')
        else:write(objective_path,result['objectiveEvaluation'])
        with store.lock():
            s=read(store.path);current=next(x for x in s['operations'] if x['id']==op['id']);card=next(x for x in s['drafts'] if x['id']==d['id'])
            if current['status']=='running':
                current.update(status='completed',progress=op['action']+' completed',finishedAt=time.time());card.update(status='revealed' if result['revealed'] else 'sealed',record=run.name,result=result);store.event(s,{'action':op['action']+'-completed','draftID':d['id']})
    except Exception as error:
        with store.lock():
            s=read(store.path);current=next(x for x in s['operations'] if x['id']==op['id']);card=next(x for x in s['drafts'] if x['id']==d['id'])
            if current['status']=='running':current.update(status='failed',progress=str(error));card['status']='interrupted';store.event(s,{'action':'failed','draftID':d['id'],'error':str(error)})
        raise

if __name__=='__main__':worker(Path(sys.argv[2]),Path(sys.argv[3]))
