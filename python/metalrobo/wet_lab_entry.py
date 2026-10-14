"""Discover durable laboratories and reconnect without reusing stale credentials."""
import json, os, socket, subprocess, sys, time, urllib.request
from pathlib import Path

HOME = Path.home()/'.numi/laboratories'
CONNECTION = Path(os.environ.get('NUMI_WET_LAB_CONNECTION', str(Path.home()/'.numi/wet-lab-active.json')))

def laboratories():
    result=[]
    if HOME.exists():
        for folder in HOME.iterdir():
            config=folder/'restore.json'
            if folder.is_dir() and config.is_file():
                try:
                    item=json.loads(config.read_text())
                    result.append({'name':folder.name,'workspace':item['workspace'],'modified':config.stat().st_mtime})
                except (ValueError,KeyError,OSError):continue
    return sorted(result,key=lambda x:x['modified'],reverse=True)

def readiness(connection=CONNECTION):
    try:
        active=json.loads(connection.read_text())
        from urllib.parse import urlsplit
        url=urlsplit(active['url'])
        if url.scheme!='http' or url.hostname!='127.0.0.1' or not url.port or url.path or url.query or url.fragment or url.username:raise ValueError('Invalid local service URL')
        req=urllib.request.Request(active['url']+'/api/health',headers={'X-Wet-Lab-Token':active['token']})
        with urllib.request.urlopen(req,timeout=1) as response:health=json.load(response)
        if health['pid']!=active['pid']:raise ValueError('Service identity differs from discovery record')
        return active,health
    except (OSError,ValueError,KeyError):return None,None

def open_laboratory(name=None,port=None,foreground=False):
    candidates=laboratories()
    if not name:
        active,health=readiness()
        if active:return {'status':'reconnected','url':active['url'],'pid':active['pid'],'workspace':health['workspace'],'credentials':'private'}
        try:
            saved=json.loads(CONNECTION.read_text()).get('workspace')
            name=next((x['name'] for x in candidates if x['workspace']==saved),None)
        except (OSError,ValueError):pass
        name=name or (candidates[0]['name'] if candidates else None)
    if not name or not any(x['name']==name for x in candidates):raise ValueError('No retained laboratory found. Restore a laboratory with numi wet-lab restore, then open it by name.')
    item=json.loads((HOME/name/'restore.json').read_text());config=item['configuration']
    # Use installed current presentation; scientific owners stay bound to the retained lab.
    service=Path(__file__).with_name('virtual_wet_lab.py')
    for path,label in [(service,'installed service'),(Path(config['vivoRoot'])/'Tools/VirtualWetLab/laboratory.py','scientific owner'),*[(Path(p),'assay '+aid) for aid,p in config['assays'].items()]]:
        if not path.is_file():raise ValueError('Missing '+label+': '+str(path)+'. Restore its verified dependency; no fallback model is selected.')
    active,health=readiness()
    if active and health['workspace']==str(Path(item['workspace']).resolve()):return {'status':'reconnected','url':active['url'],'pid':active['pid'],'name':name}
    # OS-selected loopback port avoids collisions with stale or unrelated listeners.
    if port is None:
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    argv=[sys.executable,str(service),'--vivo-root',config['vivoRoot'],'--workspace',item['workspace'],'--port',str(port),'--connection-file',str(CONNECTION)]
    if config.get('binary'):argv+=['--binary',config['binary']]
    for path in config['assays'].values():argv+=['--assay',path]
    if foreground:os.execv(sys.executable,argv)
    log=Path(item['workspace'])/'shared/service.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:process=subprocess.Popen(argv,stdout=stream,stderr=stream,start_new_session=True)
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        if process.poll() is not None:raise ValueError('Laboratory startup failed. Inspect '+str(log)+'; restore missing dependencies or retry with another port.')
        active,health=readiness()
        if active and active['pid']==process.pid:return {'name':name,'url':active['url'],'pid':process.pid,'status':'running','workspace':health['workspace'],'credentials':'fresh and private'}
        time.sleep(.1)
    raise ValueError('Laboratory readiness timed out. Inspect '+str(log))
