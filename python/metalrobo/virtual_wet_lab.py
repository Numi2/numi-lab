#!/usr/bin/env python3
"""NumiLab workspace for source-backed NumiVivo assays (local, single user)."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vivo-root', type=Path, default=os.environ.get('NUMIVIVO_ROOT'),
                        help='NumiVivo checkout containing Tools/VirtualWetLab')
    parser.add_argument('--binary', type=Path, default=os.environ.get('NUMIVIVO_BINARY'), help='Native numivivo executable')
    parser.add_argument('--assay', type=Path, required=True, help='Prepared source-bound assay.json')
    parser.add_argument('--workspace', type=Path, default=Path.cwd() / '.numi/virtual-wet-lab')
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--catalog', action='store_true', help='Print eligible specimens and exit')
    args = parser.parse_args()
    if not args.vivo_root or not args.binary:
        parser.error('Set --vivo-root and --binary (or NUMIVIVO_ROOT and NUMIVIVO_BINARY)')
    spec = importlib.util.spec_from_file_location('vivo_wetlab', args.vivo_root / 'Tools/VirtualWetLab/wetlab.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    catalog = owner.catalog(args.assay)
    if args.catalog:
        print(json.dumps(catalog, indent=2)); return
    args.workspace = args.workspace.resolve(); args.workspace.mkdir(parents=True, exist_ok=True)
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
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'nonce-" + token + "'; style-src 'unsafe-inline'; frame-ancestors 'none'")
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
            path = urlsplit(self.path).path
            try:
                if path == '/': return self.respond(200, html, 'text/html')
                if path == '/favicon.ico': return self.respond(204, '')
                if self.headers.get('X-Wet-Lab-Token') != token:
                    return self.respond(403, {'error': 'Workspace token required'})
                if path == '/api/catalog': return self.respond(200, owner.catalog(args.assay))
                if path == '/api/experiments':
                    rows = []
                    for folder in sorted(args.workspace.iterdir(), key=lambda p: p.name):
                        if folder.is_dir() and re.fullmatch('[0-9a-f]{32}', folder.name):
                            try:
                                reg = owner.check_seal(folder)
                                rows.append({'id': folder.name, 'donor': reg['donor'], 'createdAt': reg['createdAt'],
                                             'revealed': (folder / 'comparison.json').exists(), 'status': 'sealed'})
                            except Exception as error:
                                rows.append({'id': folder.name, 'status': 'failed or invalid', 'error': str(error)})
                    return self.respond(200, rows)
                if path.startswith('/api/experiments/'):
                    return self.respond(200, owner.summary(self.run_path(path.split('/')[-1])))
                self.respond(404, {'error': 'Unknown route'})
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
                if not 0 < length <= 8192: raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(length))
                if self.path == '/api/predict':
                    run = owner.predict(args.assay, args.binary, args.workspace,
                                        body['donor'], body['hours'], body['intervention'])
                    return self.respond(201, owner.summary(run))
                if self.path in ('/api/reveal', '/api/verify'):
                    run = self.run_path(body['id'])
                    if self.path == '/api/reveal':
                        owner.reveal(run, args.binary)
                        return self.respond(200, owner.summary(run))
                    return self.respond(200, owner.verify(run, args.binary))
                self.respond(404, {'error': 'Unknown route'})
            except Exception as error:
                self.respond(400, {'error': str(error)})

    server = HTTPServer(('127.0.0.1', args.port), Handler)
    print('Virtual Wet Lab: http://127.0.0.1:' + str(server.server_port), flush=True)
    print('Experiment records: ' + str(args.workspace), flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__':
    main()
