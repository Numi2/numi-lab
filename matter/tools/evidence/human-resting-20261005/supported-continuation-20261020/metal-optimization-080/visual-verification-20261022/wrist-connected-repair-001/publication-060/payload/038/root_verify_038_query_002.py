import pathlib,csv,json,numpy as np,hashlib,collections
e=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");d=e/'bone-clearance-fit-038-candidate-query/run-002';s=d/'source-preflight'
def rd(p):
 with p.open()as f:return list(csv.DictReader(f))
a=np.load(e/'bone-clearance-fit-037-componentwise-035affine-002/candidate.npz')
over=rd(d/'candidate-node-override.csv');nodes=rd(s/'all-bank-containment.csv.nodes.csv');pairs=rd(s/'all-bank-containment.csv');banks=rd(s/'all-bank-containment.csv.banks.csv')
assert len(over)==len(nodes)==7233 and len(banks)==185
for i,r in enumerate(over):
 assert int(r['global_node'])==int(a['global_node_ids'][i]) and int(r['layer'])*2411+int(r['row'])==i
 assert np.array_equal(np.asarray([r[k+'_m']for k in 'xyz'],np.float32),a['points_world_f32'][i])
fields=['aabb_proven_outside_banks','parity_outside_banks','inside_banks','ambiguous_banks','invalid_banks']
for r in nodes:assert sum(int(r[k])for k in fields)==185
tot={k:sum(int(r[k])for r in nodes)for k in fields}
assert sum(tot.values())==7233*185 and len(pairs)==2109
assert len({(r['local_node'],r['bank_body'],r['bank_stable_id'])for r in pairs})==len(pairs)
inside=[r for r in pairs if r['classification']=='inside']
assert len(inside)==1 and int(inside[0]['global_node'])==7434 and inside[0]['bank_body']=='23' and inside[0]['bank_stable_id']=='32'
assert hashlib.sha256((s/'all-bank-containment.csv').read_bytes()).hexdigest()=='a8d7e197db58019cb16f6f89d0ccb63f6e4f7e6f2fc3f6495fd4ee07c1d7a281'
result={'status':'CONFIRMED_ONE_REMAINING_NODE_BONE_PENETRATION','candidate_override_all_f32_positions_exact':True,'pairs':7233*185,'coverage':tot,'inside_witness':inside[0],'not_triangle_tet_or_physical_qualification':True}
(e/'root-review-038-query-002.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

