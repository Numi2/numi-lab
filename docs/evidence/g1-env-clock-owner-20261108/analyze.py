import collections, csv, json, struct
p = collections.defaultdict(dict)
with open('observed-snapshot.tsv', newline='') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        p[(r['arm'],int(r['phase']),r['array'])][int(r['index'])]=int(r['bits'])
h = collections.defaultdict(dict)
for line in open('observed.stderr'):
    t=line.split()
    if t and t[0]=='actorbits': h[(t[1],int(t[2]),int(t[3]))][int(t[4])]=int(t[5])
out={}
for n,stride in [('track',48),('frameDepth',144),('frameIdentity',576),('frameValidity',144),('camera',16)]:
    for e in range(2):
        a=p[('failure',2,n)];b=p[('failure',3,n)];c=p[('control',3,n)];d=p[('failure',4,n)];ii=range(e*stride,(e+1)*stride)
        out[f'{n}_env{e}']={'failure_step2_to3':sum(a[i]!=b[i] for i in ii),'failure_vs_control_step3':sum(b[i]!=c[i] for i in ii),'refusal_step3_to4':sum(b[i]!=d[i] for i in ii)}
for e in range(2):
    a=h[('failure',2,e)];b=h[('failure',3,e)];c=h[('control',3,e)]
    out[f'actor_env{e}']={'step2_to3':sum(a[i]!=b[i] for i in a),'failure_vs_control_step3':sum(b[i]!=c[i] for i in b)}
    out[f'visible_env{e}']=struct.unpack('<f',struct.pack('<I',p[('failure',2,'track')][e*48+43]))[0]
out['observed_vs_unobserved_text_equal']=open('observed.stdout','rb').read()==open('unobserved.stdout','rb').read()
print(json.dumps(out,indent=2))
