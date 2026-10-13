#!/usr/bin/env python3
"""Compare existing accepted native geometry with optional anatomy parents absent."""
from pathlib import Path
import argparse,gc,hashlib,json,mmap,struct,sys
import numpy as np
sys.path.insert(0,'/Users/n/numi-human-left-pq-publication-1344/src')
from numilab_human import common_atlas_skin_clearance as ca
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(v,m):
    if not v:raise ValueError(m)
def keys(p):
    with p.open('rb') as f: mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
    h=ca._PACK_HEADER.unpack_from(mm,0)
    ds=[ca._PACK_DIRECTORY.unpack_from(mm,ca._PACK_HEADER.size+i*ca._PACK_DIRECTORY.size) for i in range(h[2])]
    s=next(x for x in ds if x[0]==4)
    ks={tuple(struct.unpack_from('<2I',mm,s[2]+i*64+16)) for i in range(s[4])}
    mm.close();return ks
def main():
    ap=argparse.ArgumentParser()
    for k in ('control','candidate','output'):ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();req(not a.output.exists(),'preserve report')
    decls=[];pins={}
    def pin(p):pins[str(p.resolve())]=sha(p)
    for d in (a.control,a.candidate):
        ex=json.loads((d/'baseline/execution.json').read_text())
        req(ex['returncode']==0 and not ex['timed_out'] and not ex['changed_inputs'],'native failed')
        req(ex['native_argv_matches_prepared_cli_preview'] and ex['native_environment_matches_prepared_cli_preview'],'invocation mismatch')
        decls.append(json.loads((d/'baseline/run-declaration.json').read_text()))
        for p in (d/'baseline/execution.json',d/'baseline/run-declaration.json',d/'baseline/native-run/resting-coupled.csv'):pin(p)
    req(decls[0]['capture_steps']==decls[1]['capture_steps'],'different sample grid')
    bs=[d['owner_cli_preview']['native_argv'][0] for d in decls]
    req([sha(b) for b in bs]==['d51d62bc1baa928bbd712eebd5fe536a6bbace39ea45ca9f54bbc1d052da4771','72cdb21875a1929b60f609909cc6e79f183d65c113b572d6a49a7329a4517d7f'],'different regression binary pair')
    phys=[d/'baseline/native-run/resting-coupled.csv' for d in (a.control,a.candidate)]
    req(phys[0].read_bytes()==phys[1].read_bytes(),'physiology changed')
    pin(Path(__file__));pin(Path(ca.__file__))
    r={'complete':False,'regression_binary_sha256':[sha(b) for b in bs],'physiology_byte_exact':True,
       'scope':'All native surface positions and body poses across eight matched captures; optional anatomy parent bindings absent in both scenes. Forearm and later axillary vein defects remain outside anatomical qualification.',
       'captures':{},'pins':pins}
    for step in decls[0]['capture_steps']:
        ps=[d/'baseline/native-run/accepted-geometry'/('step-'+str(step)+'.mrvpack') for d in (a.control,a.candidate)]
        receipts=[]
        for p in ps:
            pin(p);rp=p.with_suffix('.receipt.json');pin(rp)
            rec=json.loads(rp.read_text());receipts.append(rec)
            req(rec['physical_endpoint']=='accepted' and rec['pack_file_sha256']==sha(p),'invalid accepted capture')
        req(receipts[0]['accepted_body_poses']==receipts[1]['accepted_body_poses'],'physical poses changed')
        ks=[keys(p) for p in ps]
        req(ks[0]==ks[1] and (51010,22) not in ks[0],'undeclared surface inventory difference')
        requested=ks[1]
        packs=[ca._pack_surfaces(p,requested) for p in ps]
        x,s,_=packs[0];y,t,_=packs[1]
        req(set(s)==requested and set(t)==requested,'missing compared surface')
        for k in requested:
            f=np.asarray(s[k]['faces']);g=np.asarray(t[k]['faces'])
            req(f.shape==g.shape and x[f].tobytes()==y[g].tobytes(),'position/topology mismatch '+str(k))
        r['captures'][str(step)]={'body_poses_exact':True,'unchanged_surface_count':len(requested),
          'surface_inventory_exact':True}
        a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'step':step,'unchanged_surface_count':len(requested)}),flush=True)
        del packs,x,y,s,t;gc.collect()
    req(all(sha(p)==h for p,h in pins.items()),'audit input drift')
    r['complete']=True;a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
