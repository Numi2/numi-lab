#!/usr/bin/env python3
"""NumiLab workspace for source-backed NumiVivo assays (local, single user)."""
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vivo-root', type=Path, default=os.environ.get('NUMIVIVO_ROOT'),
                        help='NumiVivo checkout containing Tools/VirtualWetLab')
    parser.add_argument('--binary', type=Path, default=os.environ.get('NUMIVIVO_BINARY'), help='Native numivivo executable')
    parser.add_argument('--assay', type=Path, action='append', required=True, help='Prepared assay.json; repeat for another adapter')
    parser.add_argument('--workspace', type=Path, default=Path.cwd() / '.numi/virtual-wet-lab')
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--catalog', action='store_true', help='Print eligible specimens and exit')
    parser.add_argument('--design-campaign', type=Path, help='Sealed NumiVivo target-aware intervention campaign')
    parser.add_argument('--receiving-campaign', type=Path, help='Qualified explicit-receiver development artifacts')
    args = parser.parse_args()
    if not args.vivo_root:
        parser.error('Set --vivo-root (or NUMIVIVO_ROOT)')
    sys.path.insert(0, str(args.vivo_root / 'Tools/VirtualWetLab'))
    import laboratory as owner
    runtime = {'binary': args.binary}
    configs = {owner.rna.read(p)['id']: p for p in args.assay}
    if len(configs) != len(args.assay): parser.error('Duplicate assay IDs')
    if not args.binary and any(owner.adapter_for_config(p).family == 'cell-response' for p in args.assay):
        parser.error('RNA assays require --binary (or NUMIVIVO_BINARY)')
    def catalog_for(config): return owner.adapter_for_config(config).catalog(config)
    def present(adapter, run):
        result=adapter.summary(run)
        if adapter.family=='spatial-tissue' and result['revealed']:
            result['observationFields']=owner.rna.read(run/'observations.json')
        return result
    catalog = catalog_for(args.assay[0])
    if args.catalog:
        print(json.dumps([catalog_for(p) for p in args.assay], indent=2)); return
    args.workspace = args.workspace.resolve(); args.workspace.mkdir(parents=True, exist_ok=True)
    from wet_lab_shared import SharedWorkspace, Conflict, write as shared_write
    shared = SharedWorkspace(args.workspace, {'vivoRoot': str(args.vivo_root.resolve()), 'binary': str(args.binary) if args.binary else None, 'assays': {k: str(v.resolve()) for k,v in configs.items()}})
    shared.reconcile()
    token = secrets.token_urlsafe(32)
    html = (Path(__file__).parent / 'wet_lab/index.html').read_text().replace('__TOKEN__', token)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, kind='application/json'):
            encoded = body.encode() if isinstance(body, str) else json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', kind + '; charset=utf-8')
            self.send_header('Content-Length', str(len(encoded)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'nonce-" + token + "'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
            self.end_headers(); self.wfile.write(encoded)

        def valid_host(self):
            return self.headers.get('Host') == '127.0.0.1:' + str(self.server.server_port)

        def run_path(self, run_id):
            if not re.fullmatch('[0-9a-f]{32}', run_id):
                raise ValueError('Invalid experiment ID')
            path = args.workspace / run_id
            if path.is_symlink() or not path.is_dir(): raise ValueError('Experiment not found')
            return path

        def do_GET(self):
            if not self.valid_host(): return self.respond(403, {'error': 'Use the local workspace URL'})
            parsed = urlsplit(self.path); path = parsed.path
            try:
                if path == '/': return self.respond(200, html, 'text/html')
                if path in ('/app.js','/tissue_index.js','/learned.js','/design.js','/shared.js','/style.css'):
                    return self.respond(200, (Path(__file__).parent / 'wet_lab' / path[1:]).read_text(), 'text/javascript' if path.endswith('.js') else 'text/css')
                if path == '/favicon.ico': return self.respond(204, '')
                if self.headers.get('X-Wet-Lab-Token') != token:
                    return self.respond(403, {'error': 'Workspace token required'})
                if path == '/api/shared': return self.respond(200, shared.state())
                if path == '/api/templates':
                    folder=args.workspace/'templates'
                    return self.respond(200, [owner.rna.read(p) for p in sorted(folder.glob('*.json'))] if folder.exists() else [])
                if path == '/api/design':
                    if not args.design_campaign: return self.respond(200, {'available': False})
                    import intervention_design
                    return self.respond(200, {'available': True, **intervention_design.catalog(args.design_campaign)})
                if path == '/api/design/history':
                    rows = []
                    if args.design_campaign:
                        for folder in sorted(args.workspace.glob('campaign-*')):
                            if folder.is_symlink() or not re.fullmatch('campaign-[0-9a-f]{32}', folder.name) or not (folder/'registration.json').is_file(): continue
                            reg = owner.rna.read(folder/'registration.json')
                            if reg.get('kind') != 'intervention-design' or Path(reg['campaign']).resolve() != args.design_campaign.resolve(): continue
                            rows.append({'id': folder.name, 'createdAt': reg['createdAt'],
                                         'selection': reg['selection'], 'revealed': (folder/'comparison.json').exists()})
                    return self.respond(200, rows)
                if path == '/api/receivers':
                    if not args.receiving_campaign:return self.respond(200,{'available':False})
                    from receiving_inspection import inspect
                    return self.respond(200,{'available':True,**inspect(args.receiving_campaign)})
                if path == '/api/qualification':
                    from qualification import ARC_2026
                    return self.respond(200, ARC_2026)
                if path == '/api/assays': return self.respond(200, [catalog_for(p) for p in args.assay])
                if path == '/api/catalog':
                    assay_id = parse_qs(parsed.query).get('assay', [catalog['id']])[0]
                    return self.respond(200, catalog_for(configs[assay_id]))
                if path == '/api/experiments':
                    rows = []
                    for folder in sorted(args.workspace.iterdir(), key=lambda p: p.name):
                        if folder.is_dir() and re.fullmatch('[0-9a-f]{32}', folder.name):
                            try:
                                adapter = owner.adapter_for_run(folder)
                                details = adapter.summary(folder); reg = details['registration']
                                rows.append({'id': folder.name, 'donor': reg.get('donor', reg.get('specimen')), 'family': adapter.family, 'assayID': reg['assay']['id'], 'createdAt': reg['createdAt'],
                                             'revealed': (folder / 'comparison.json').exists(), 'status': 'sealed'})
                            except Exception as error:
                                rows.append({'id': folder.name, 'status': 'failed or invalid', 'error': str(error)})
                    return self.respond(200, rows)
                if path.startswith('/api/experiments/'):
                    return self.respond(200, present(owner.adapter_for_run(self.run_path(path.split('/')[-1])),self.run_path(path.split('/')[-1])))
                self.respond(404, {'error': 'Unknown route'})
            except Conflict as error:
                self.respond(409, {'error': str(error), 'revision': shared.state()['revision']})
            except Exception as error:
                self.respond(400, {'error': str(error)})

        def do_POST(self):
            if not self.valid_host() or self.headers.get('X-Wet-Lab-Token') != token:
                return self.respond(403, {'error': 'Workspace token required'})
            origin = self.headers.get('Origin')
            if origin and origin != 'http://127.0.0.1:' + str(self.server.server_port):
                return self.respond(403, {'error': 'Cross-origin request rejected'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 65536: raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(length))
                if self.path == '/api/shared': return self.respond(200, shared.mutate(body))
                if self.path == '/api/receivers':
                    if not args.receiving_campaign:raise ValueError('Receiver experiment unavailable')
                    from receiving_inspection import inspect
                    return self.respond(200,inspect(args.receiving_campaign,body['gene'],body['group']))
                if self.path.startswith('/api/design/'):
                    if not args.design_campaign: raise ValueError('No qualified design campaign is installed')
                    import intervention_design as design
                    action=self.path.rsplit('/',1)[-1]
                    if action=='preview': return self.respond(200,design.preview(args.design_campaign,body['selection']))
                    if action=='seal': return self.respond(201,design.seal(args.design_campaign,args.workspace,body['selection']))
                    identifier=body.get('id','')
                    if not re.fullmatch('campaign-[0-9a-f]{32}',identifier): raise ValueError('Invalid design campaign')
                    root=args.workspace/identifier
                    if root.is_symlink() or not root.is_dir(): raise ValueError('Campaign not found')
                    if action=='reveal': return self.respond(200,design.reveal(root))
                    if action=='verify': return self.respond(200,design.verify(root))
                    if action=='open':
                        design.check(root)
                        return self.respond(200,{'id':identifier,'prediction':owner.rna.read(root/'prediction.json'),'revealed':(root/'comparison.json').exists(),'comparison':owner.rna.read(root/'comparison.json') if (root/'comparison.json').exists() else None})
                    raise ValueError('Unknown design action')
                if self.path == '/api/templates':
                    if set(body) != {'name','assayID','selection'} or not isinstance(body['name'],str) or not 1<=len(body['name'])<=100: raise ValueError('Template name and typed selection required')
                    config=configs[body['assayID']]; adapter=owner.adapter_for_config(config)
                    if not hasattr(adapter,'compile'): raise ValueError('Templates require a typed-plan assay')
                    adapter.compile(config,body['selection'])
                    folder=args.workspace/'templates';folder.mkdir(exist_ok=True)
                    identifier=secrets.token_hex(16);saved={**body,'id':identifier};owner.rna.write(folder/(identifier+'.json'),saved)
                    return self.respond(201,saved)
                if self.path == '/api/compile':
                    config=configs[body['assayID']]; adapter=owner.adapter_for_config(config)
                    if not hasattr(adapter,'compile'): raise ValueError('This legacy assay uses its registered protocol directly')
                    return self.respond(200,adapter.compile(config,body['selection']))
                if self.path == '/api/propose':
                    from experiment_campaign import propose_next
                    run=self.run_path(body['id'])
                    return self.respond(200,propose_next(owner.adapter_for_run(run).summary(run)))
                if self.path == '/api/campaign':
                    from experiment_campaign import register_and_predict
                    root=register_and_predict(configs[body['assayID']],runtime,args.workspace,body['request'])
                    return self.respond(201,{'id':root.name,'registration':owner.rna.read(root/'registration.json'),'seal':owner.rna.read(root/'prediction-seal.json')})
                if self.path in ('/api/campaign/reveal','/api/campaign/verify'):
                    from experiment_campaign import reveal,verify
                    identifier=body['id']
                    if not re.fullmatch('campaign-[0-9a-f]{32}',identifier): raise ValueError('Invalid campaign')
                    root=args.workspace/identifier
                    if root.is_symlink() or not root.is_dir(): raise ValueError('Campaign not found')
                    return self.respond(200,(reveal if self.path.endswith('reveal') else verify)(root,runtime))
                if self.path in ('/api/specimen','/api/spatial-feature'):
                    config=configs[body['assayID']]; adapter=owner.adapter_for_config(config)
                    if adapter.family!='learned-spatial-response': raise ValueError('Spatial response assay required')
                    if self.path=='/api/specimen': return self.respond(200,adapter.geometry(config,body['specimen']))
                    run=self.run_path(body['id']) if body.get('id') else None
                    return self.respond(200,adapter.feature(config,body['specimen'],body['gene'],run))
                if self.path == '/api/predict':
                    config = configs[body['assayID']]; adapter = owner.adapter_for_config(config)
                    run = adapter.predict(config, runtime, args.workspace, body['selection'])
                    return self.respond(201, present(adapter,run))
                if self.path in ('/api/reveal', '/api/verify'):
                    run = self.run_path(body['id'])
                    adapter = owner.adapter_for_run(run)
                    if self.path == '/api/reveal':
                        adapter.reveal(run, runtime)
                        return self.respond(200, present(adapter,run))
                    return self.respond(200, adapter.verify(run, runtime))
                self.respond(404, {'error': 'Unknown route'})
            except Conflict as error:
                self.respond(409, {'error': str(error), 'revision': shared.state()['revision']})
            except Exception as error:
                self.respond(400, {'error': str(error)})

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    active = Path.home()/'.numi/wet-lab-active.json'
    active.parent.mkdir(exist_ok=True)
    shared_write(active, {'url': 'http://127.0.0.1:'+str(server.server_port), 'token': token, 'pid': os.getpid()})
    active.chmod(0o600)
    print('Virtual Wet Lab: http://127.0.0.1:' + str(server.server_port), flush=True)
    print('Experiment records: ' + str(args.workspace), flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__':
    main()
