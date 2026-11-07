import collections,csv,json
p=collections.defaultdict(dict)
with open('selection.tsv',newline='') as f:
    for r in csv.DictReader(f,delimiter='\t'):
        p[(r['frame'],r['array'])][int(r['index'])]=int(r['bits'])
out={}
for name,stride in [('depth',144),('identity',576),('validity',144),('camera',16),('metadata',16)]:
    held=p[('held',name)];fresh=p[('fresh',name)]
    for e in range(2):
        def source(index): return index if name!='metadata' else index%stride
        ids=range(e*stride,(e+1)*stride)
        out[f'{name}_env{e}']={'fresh_vs_held':sum(held[source(i)]!=fresh[source(i)] for i in ids)}
        for scenario in range(2):
            actual=p[(f'scenario{scenario}',name)]
            ref=fresh if e==1 and scenario==1 else held
            out[f'{name}_env{e}'][f'scenario{scenario}_mismatches']=sum(actual[i]!=ref[source(i)] for i in ids)
print(json.dumps(out,indent=2))
