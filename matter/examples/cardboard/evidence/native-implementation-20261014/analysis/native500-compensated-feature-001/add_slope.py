import runpy, contextlib, io, json
from decimal import localcontext
from pathlib import Path
base=Path('/Users/n/cardboard-implementation-20261014-host-macmini/analysis/native500-compensated-feature-001')
ns={}
with contextlib.redirect_stdout(io.StringIO()):
    ns=runpy.run_path(str(base/'energy_armijo_analysis.py'))
with localcontext() as ctx:
    ctx.prec=90
    out=[]
    for row in ns['rows']:
        i=row['row']
        pts,tri,native_sep,native_E,sep,bary,branch=ns['endpoint'](i,'current')
        closest,bary,branch=ns['closest'](pts[0],*tri)
        r=ns['sub'](pts[0],closest)
        dirs=[]
        for node in ns['nodes']:
            dirs.append(tuple(ns['df'](ns['f32'](ns['timestep']*ns['flt'](ns['cor'],node*16+4*j))) for j in range(3)))
        w=bary[2]
        rel=ns['sub'](ns['sub'](dirs[0],dirs[2]),ns['scale'](w,ns['sub'](dirs[3],dirs[2])))
        ds=ns['dot'](r,rel)/sep
        dE=ns['barrier_derivative'](sep,ns['thick'],ns['stiff'])*ds
        out.append({'row':i,'branch':branch,'bary':list(map(str,bary)),'sep_m':str(sep),'dsep_dalpha':str(ds),'denergy_dalpha_J':str(dE),'native_directional_derivative_J':row['source_float_directional_derivative_transcription']})
    (base/'exact-dd-slope.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps(out,indent=2))
