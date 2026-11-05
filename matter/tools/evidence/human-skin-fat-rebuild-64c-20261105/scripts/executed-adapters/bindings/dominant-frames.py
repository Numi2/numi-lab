from pathlib import Path
import csv,json,numpy as np,hashlib,io
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root');H=S.parent;E=Path('/Users/numi/Work/human-integrated-gaps-20261104/joint-boundary-rebuild-64c/compiler/volume-export');out=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
vol=Path('/Users/numi/Work/human-integrated-gaps-20261104/joint-boundary-rebuild-64c/compiler/rich433/candidate-current-typed-volume.npz');assert sha(vol)=='0152c7b15febfc81f4d9ce995820a80438204635deb425193212409e31c5c323';z=np.load(vol,allow_pickle=False);weights=z['full86_weights'];frames=list(csv.DictReader((E/'registered/donor-frames.csv').open()));lanes=np.array([int(r['donor']) for r in frames]);assert len(lanes)==86
frame={}
for f in frames:
 body=int(f['donor']);R=np.array([[float(f[f'current_R{a}{b}']) for b in range(3)] for a in range(3)]);pos=np.array([float(f['current_'+a]) for a in 'xyz']);frame[body]=(np.linalg.inv(R),pos,int(f['binding']))
inputs={str(vol):sha(vol),str(E/'registered/donor-frames.csv'):sha(E/'registered/donor-frames.csv')};outputs={};changes={};witnesses=[]
def rewrite_csv(src,dst,update):
 inputs[str(src)]=sha(src);count=0
 with src.open(newline='') as f,dst.open('w',newline='') as g:
  head=f.readline();g.write(head);columns=next(csv.reader([head]))
  for line in f:
   fields=next(csv.reader([line]));assert len(fields)==len(columns);row=dict(zip(columns,fields));new=update(row)
   if new is None:g.write(line)
   else:
    assert set(new)==set(row);b=io.StringIO(newline='');csv.DictWriter(b,fieldnames=columns,lineterminator='\n').writerow(new);g.write(b.getvalue());count+=1
 outputs[str(dst)]={'sha256':sha(dst),'changed_rows':count}
 return count
for role in 'OID':
 source=E/f'roles/{role}-source.json';inputs[str(source)]=sha(source);doc=json.loads(source.read_text());vertices=doc['vertices'];local_changes={}
 for row,v in enumerate(vertices):
  node=v['child_volume_node_row'];w=weights[node];lane=int(np.argmax(w));body=int(lanes[lane]);oldbody=v['registered_frame_owner_body']
  if body==oldbody:continue
  x=np.array(v['current_world_m']);M,pos,binding=frame[body];assert binding==lane;local=M@(x-pos);assert np.max(np.abs(np.linalg.solve(M,local)+pos-x))<1e-14
  data={'donor':body,'binding':binding,'local':local.tolist(),'matrix':M.tolist()};local_changes[row]=data;assert node not in changes or changes[node]==data;changes[node]=data
  witnesses.append({'layer':role,'compact_row':row,'source_vertex_row':v['source_vertex_row'],'volume_node_row':node,'old_frame_owner':oldbody,'dominant_frame_owner':body,'dominant_binding':lane,'dominant_weight':float(w[lane]),'old_owner_weight':float(w[np.flatnonzero(lanes==oldbody)[0]]),'position_m':x.tolist(),'new_donor_local_m':local.tolist()});v['registered_frame_owner_body']=body;v['registered_donor_local_m']=local.tolist()
 def update(row):
  j=int(row['vertex'])
  if j not in local_changes:return None
  a=local_changes[j];v=dict(row);v['donor']=str(a['donor'])
  for k,x in zip(['lx','ly','lz'],a['local']):v[k]=format(x,'.17g')
  for i in range(3):
   for j in range(3):v[f'm{i}{j}']=format(a['matrix'][i][j],'.17g')
  for key in ['wx','wy','wz','nx','ny','nz','source_vertex_row']:assert v[key]==row[key]
  return v
 dst=out/f'registered-{role}.csv';assert rewrite_csv(E/f'registered/registered-{role}.csv',dst,update)==len(local_changes)
 doc['derivation']['dominant_frame_registration']={'method':'recompute body-local registration in existing dominant full86 current body frame; unchanged world geometry, weights, source IDs, cells and mass','original_role_asset_sha256':inputs[str(source)],'changed_rows':len(local_changes)}
 dst=out/f'{role}-source.json';dst.write_text(json.dumps(doc,separators=(',',':'))+'\n');outputs[str(dst)]={'sha256':sha(dst),'changed_rows':len(local_changes)}
 del doc,vertices

def update_joint(row):
 node=int(row['joint_node_id'])-1
 if node not in changes:return None
 a=changes[node];v=dict(row);v['donor_body']=str(a['donor']);v['donor_binding']=str(a['binding'])
 for k,x in zip(['donor_local_x_m','donor_local_y_m','donor_local_z_m'],a['local']):v[k]=format(x,'.17g')
 assert all(v[k]==row[k] for k in row if k not in ['donor_body','donor_binding','donor_local_x_m','donor_local_y_m','donor_local_z_m'])
 return v
count=rewrite_csv(E/'volume/combined-node-map.csv',out/'combined-node-map.csv',update_joint);assert count>=len(changes)
receipt={'status':'PASS_CURRENT_DOMINANT_FRAME_REGISTRATION_ONLY','physical_admission':False,'native_steps':0,'changed_boundary_nodes':len(changes),'changed_combined_csv_rows':count,'world_geometry_changed':False,'source_ids_changed':False,'full86_weights_changed':False,'physical_cell_semantics_changed':False,'mass_allocation_changed':False,'historical_registration_changed':False,'inputs':inputs,'outputs':outputs,'witnesses':witnesses}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ['inputs','witnesses']},indent=2))
