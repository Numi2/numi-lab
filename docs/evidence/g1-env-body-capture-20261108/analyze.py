import collections,csv,json
p=collections.defaultdict(dict)
with open('snapshots.tsv',newline='') as f:
    for r in csv.DictReader(f,delimiter='\t'):
        p[(r['phase'],r['array'])][int(r['index'])]=int(r['bits'])
out={}
for name,stride in [('depth',144),('identity',576),('validity',144),('camera',16),('track',48),('actor',140)]:
    for e in range(2):
        before=p[('before',name)];after=p[('after',name)];ref=p.get(('reference',name),{});ii=range(e*stride,(e+1)*stride)
        out[f'{name}_env{e}']={'after_vs_before':sum(before[i]!=after[i] for i in ii),'after_vs_reference':sum(after[i]!=ref[i] for i in ii) if ref else None}
a=p[('captured','body')];b=p[('later','body')];stride=len(a)//2
for e in range(2):out[f'body_env{e}_changed_words']=sum(a[i]!=b[i] for i in range(e*stride,(e+1)*stride))
out['metadata']={phase:[p[(phase,'metadata')][i] for i in range(16)] for phase in ['before','after','reference']}
print(json.dumps(out,indent=2))
