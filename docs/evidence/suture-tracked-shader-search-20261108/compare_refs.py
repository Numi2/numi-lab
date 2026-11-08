import json, pathlib, re, subprocess
p=pathlib.Path(__file__).resolve().parent
repo='/Users/numi/numi-lab-reconciliation-20261108'
scanned=set(json.loads((p/'inventory.json').read_text())['objects'])
rows=[]
for line in (p/'remote-refs.txt').read_text().splitlines():
    oid, ref = line.split('\t')
    check=subprocess.run(['git','-C',repo,'cat-file','-e',oid+'^{commit}'])
    if check.returncode:
        rows.append({'ref':ref,'commit':oid,'missing':True});continue
    tree=subprocess.check_output(['git','-C',repo,'ls-tree','-r','-z',oid])
    objects=set();count=0
    for item in tree.split(b'\0'):
        if not item:continue
        header,path=item.split(b'\t',1)
        if re.search(rb'\.metallib(?:\.gz)?$',path,re.I):
            objects.add(header.split()[-1].decode());count+=1
    rows.append({'ref':ref,'commit':oid,'paths':count,'unique_objects':len(objects),'outside_scanned':sorted(objects-scanned)})
(p/'remote-comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
print('remote_refs',len(rows),'missing',sum(bool(x.get('missing')) for x in rows),'unscanned_objects',len({x for row in rows for x in row.get('outside_scanned',[])}))
if any(x.get('missing') or x.get('outside_scanned') for x in rows):raise SystemExit(1)
