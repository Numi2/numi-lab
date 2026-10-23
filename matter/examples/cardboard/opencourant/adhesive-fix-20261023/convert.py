from pathlib import Path
import subprocess,json,hashlib,sys
T=Path('/Users/n/cardboard-opencourant-20261023')
R=T/'adhesive-fix'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
converter=T/'runtime/OpenCourant/exec/th_to_csv_linuxa64'
for run in sorted((R/'runs').glob('*')):
    if not (run/'run.json').is_file(): continue
    for source in sorted(run.glob('*T[0-9][0-9]')):
        titles=Path(str(source)+'_TITLES')
        if not titles.is_file(): continue
        out=R/'post'/run.name/source.name[-3:]
        if out.exists(): continue
        out.mkdir(parents=True)
        (out/'probeT01').write_bytes(source.read_bytes())
        (out/'probeT01_TITLES').write_bytes(titles.read_bytes())
        cmd=[str(converter),'probeT01']
        with (out/'stdout').open('w') as stdout,(out/'stderr').open('w') as stderr:
            p=subprocess.run(cmd,cwd=out,stdout=stdout,stderr=stderr,timeout=30)
        data={'schema':'cardboard.opencourant.postprocess.v1','argv':cmd,'cwd':str(out),'exit_code':p.returncode,'input_relative_path':source.name,'input_sha256':sha(source),'input_alias':'probeT01','titles_sha256':sha(titles),'converter_sha256':sha(converter),'output':'probeT01.csv'}
        if (out/'probeT01.csv').exists():data['output_sha256']=sha(out/'probeT01.csv')
        (out/'conversion.json').write_text(json.dumps(data,indent=2)+'\n')
        print(run.name,source.name[-3:],p.returncode)
