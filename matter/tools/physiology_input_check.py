"""Admission checks only; this tool never steps physical state."""
import json, pathlib, subprocess, tempfile, sys
cli=sys.argv[1]
original=pathlib.Path(sys.argv[2]).read_text()
base=json.loads(original)
cases={
    'duplicate_root':original.replace('"schema":','"schema":"bogus","schema":',1),
    'duplicate_escaped_root':original.replace('"schema":','"sch\\u0065ma":"bogus","schema":',1),
    'duplicate_nested':original.replace('"volume_m3":','"volume_m3":42,"volume_m3":',1),
}
def variant(name, mutate):
    obj=json.loads(original);mutate(obj);cases[name]=json.dumps(obj)
variant('unsupported_law',lambda x:x.update(law='pump_v99'))
variant('unknown_field',lambda x:x.update(replenish_blood=True))
variant('boolean_scale',lambda x:x['species'][0].update(amount_scale_mol=True))
variant('duplicate_volume_owner',lambda x:x['tissue_reservoirs'][0].update(physical_volume_owner_id=x['compartments'][0]['physical_volume_owner_id']))
variant('missing_endpoint',lambda x:x['connections'][0].update(to=900))
variant('invalid_sha',lambda x:x.update(source_graph_sha256='x'*64))
variant('unsupported_qualification',lambda x:x.update(qualification='validated_human'))
variant('negative_amount',lambda x:x['compartments'][0].update(initial_species_mol=[-1]))
variant('unknown_species',lambda x:x['exchanges'][0].update(species=900))
with tempfile.TemporaryDirectory(prefix='numi-physiology-input-') as tmp:
    root=pathlib.Path(tmp)
    for name,data in [('valid',original),*cases.items()]:
        source=root/(name+'.json');output=root/(name+'.nmatterpack');source.write_text(data)
        p=subprocess.run([cli,str(source),str(output),'--timestep','0.01','--environments','2'],capture_output=True,text=True)
        if name=='valid':
            assert p.returncode==0 and output.is_file(),p.stdout+p.stderr
            print(p.stdout.strip())
        else:
            assert p.returncode!=0 and not output.exists(),name+' accepted invalid source'
            print('native_input_rejected='+name)
print('native_input_admission=pass positive_cases=1 negative_cases='+str(len(cases)))
