import gzip, hashlib, io, json, os, pathlib, re, subprocess, sys, tempfile
p = pathlib.Path(__file__).resolve().parent
root = pathlib.Path('/Users/numi/numi-lab-reconciliation-20261108')
branch = 'codex/suture-earned-continuation'
ref = subprocess.check_output(['git', '-C', str(root), 'rev-parse', branch], text=True).strip()
data = subprocess.check_output(['git', '-C', str(root), 'ls-tree', '-r', '-z', ref])
blobs = {}
for item in data.split(b'\0'):
    if not item:
        continue
    header, raw_path = item.split(b'\t', 1)
    path = raw_path.decode('utf-8', errors='surrogateescape')
    if re.fullmatch(r'.*\.metallib(?:\.gz)?', pathlib.PurePosixPath(path).name, re.I):
        mode, typ, oid = header.decode().split()
        if typ == 'blob':
            blobs.setdefault(oid, []).append(path)
(p / 'inventory.json').write_text(json.dumps({'ref': ref, 'paths': sum(map(len, blobs.values())), 'unique_blobs': len(blobs), 'objects': blobs}, indent=2) + '\n')
limit = 64 * 1024 * 1024
rows = []
with tempfile.TemporaryDirectory(prefix='numi-shader-scan-', dir=p) as scratch:
    target = pathlib.Path(scratch) / 'candidate.metallib'
    for index, (oid, paths) in enumerate(sorted(blobs.items()), 1):
        st = os.statvfs(p)
        if st.f_bavail * st.f_frsize < 4 * 1024**3:
            raise SystemExit('disk preflight failed; partial results retained')
        row = {'oid': oid, 'paths': paths}
        try:
            size = int(subprocess.check_output(['git', '-C', str(root), 'cat-file', '-s', oid], text=True))
            if size > limit:
                raise ValueError('compressed blob exceeds 64 MiB')
            content = subprocess.check_output(['git', '-C', str(root), 'cat-file', 'blob', oid])
            row['stored_sha256'] = hashlib.sha256(content).hexdigest()
            if content.startswith(b'\x1f\x8b'):
                with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
                    content = gz.read(limit + 1)
            if len(content) > limit:
                raise ValueError('decompressed blob exceeds 64 MiB')
            row.update(size=len(content), sha256=hashlib.sha256(content).hexdigest())
            target.write_bytes(content)
            result = subprocess.run([str(p / 'match')], input=str(target)+'\n', text=True, capture_output=True, check=True, timeout=60)
            row['fingerprint'] = result.stdout.split('\t')[0]
            row['match'] = row['fingerprint'] == '17848984240238310200'
        except Exception as e:
            row['error'] = str(e)
        rows.append(row)
        with (p / 'results.jsonl').open('a') as f:
            f.write(json.dumps(row, ensure_ascii=True) + '\n')
        print(f"{index}/{len(blobs)} {oid[:12]} match={row.get('match')} error={row.get('error')}", flush=True)
(p / 'summary.json').write_text(json.dumps({'ref': ref, 'total_paths': sum(map(len, blobs.values())), 'unique_blobs': len(blobs), 'matches': sum(r.get('match', False) for r in rows), 'errors': sum('error' in r for r in rows)}, indent=2)+'\n')
if any('error' in r for r in rows):
    sys.exit(1)
