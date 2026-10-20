#!/usr/bin/env python3
"""Compare frozen whole-body skin source ownership and mass partition before/after layer repair."""
from __future__ import annotations
import argparse,hashlib,json,re,subprocess
from pathlib import Path
import numpy as np

def digest(text:str)->str:return hashlib.sha256(text.encode()).hexdigest()
def block(text:str,name:str)->str:
 m=re.search(re.escape(name)+r'\s*=\s*\{',text)
 if not m:raise ValueError('missing '+name)
 ends=[x for x in (text.find('\n};',m.end()),text.find('\n}};',m.end())) if x>=0]
 if not ends:raise ValueError('unclosed '+name)
 return text[m.end():min(ends)]
def source_debits_formula(text:str):
 return {'skin':'skinLayer.volumeM3*1000.0' in text, 'fat':'fatLayer.volumeM3*900.0' in text}
def mass(text:str,name:str)->float:
 m=re.search(re.escape(name)+r'\s*=\s*([-+0-9.eE]+)\s*;',text)
 if not m:raise ValueError('missing numeric constant '+name)
 return float(m.group(1))
def outer_points(text:str):
 fb=block(text,'kWholeBodySkinDonorFrames')
 frames=np.asarray([[float(x.strip().rstrip('f')) for x in m.group(3).split(',')] for m in re.finditer(r'WholeBodySkinDonorFrameV1\{\s*\d+u,\s*\{([^{}]+)\},\s*\{([^{}]+)\},\s*\{([^{}]+)\},\s*\{([^{}]+)\}\s*\}',fb)],dtype=np.float64)
 db=block(text,'kWholeBodySkinMechanicalWorldDeltas')
 d=np.asarray([[float(x) for x in row.split(',')] for row in re.findall(r'std::array<double,3>\{([^{}]+)\}',db)],dtype=np.float64)
 return frames+d

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path.cwd());ap.add_argument('--before-revision',default='HEAD');ap.add_argument('--after-header',type=Path);ap.add_argument('--output',type=Path);args=ap.parse_args();root=args.root.resolve()
 rel=Path('apps/NumiHumanWholeBodySkinReduction.hpp');old=subprocess.run(['git','show',f'{args.before_revision}:{rel.as_posix()}'],cwd=root,check=True,capture_output=True,text=True).stdout
 newpath=(args.after_header or root/rel).resolve();new=newpath.read_text()
 names=['kWholeBodySkinSourceVertices','kWholeBodySkinFaces','kWholeBodySkinSourceFaces','kWholeBodySkinDonorFrames','kWholeBodySkinMechanicalWorldDeltas','kWholeBodySkinMechanicalDonorLocalDeltas','kWholeBodySkinOutwardDirections']
 comparisons={}
 for n in names:
  a=block(old,n);b=block(new,n);comparisons[n]={'exactly_unchanged':a==b,'before_block_sha256':digest(a),'after_block_sha256':digest(b)}
 layer_names=['kWholeBodySkinInnerWorld','kWholeBodySkinInnerLocal','kWholeBodyFatDeepWorld','kWholeBodyFatDeepLocal','kWholeBodySkinThicknessM','kWholeBodyFatThicknessM']
 layer_comparisons={}
 for n in layer_names:
  a=block(old,n);b=block(new,n);layer_comparisons[n]={'exactly_unchanged':a==b,'before_block_sha256':digest(a),'after_block_sha256':digest(b)}
 oldouter=outer_points(old);newouter=outer_points(new)
 debit_names=['kWholeBodySkinMassDebitKg','kWholeBodyFatMassDebitKg']
 volume_names=['kWholeBodySkinLayerVolumeM3','kWholeBodyFatLayerVolumeM3']
 old_vols={n:mass(old,n) for n in volume_names};new_vols={n:mass(new,n) for n in volume_names}
 old_app=subprocess.run(['git','show',f'{args.before_revision}:apps/numilab_human_myosim_visual_probe.mm'],cwd=root,check=True,capture_output=True,text=True).stdout
 new_app=(root/'apps/numilab_human_myosim_visual_probe.mm').read_text()
 old_debits={'kWholeBodySkinMassDebitKg':old_vols['kWholeBodySkinLayerVolumeM3']*1000.0,'kWholeBodyFatMassDebitKg':old_vols['kWholeBodyFatLayerVolumeM3']*900.0}
 new_debits={n:mass(new,n) for n in debit_names}
 rev=subprocess.run(['git','rev-parse',args.before_revision],cwd=root,check=True,capture_output=True,text=True).stdout.strip()
 receipt={'schema':'whole_body_skin_fat_source_preservation_v1','before':{'source':f'git {args.before_revision}:{rel.as_posix()}','revision':rev,'sha256':digest(old)},'after':{'source':str(newpath.relative_to(root)) if newpath.is_relative_to(root) else str(newpath),'sha256':digest(new)},'exterior_and_source_ownership':{'arrays':comparisons,'derived_registered_outer_plus_mechanical_delta_max_abs_difference_m':float(np.max(np.abs(oldouter-newouter))),'vertex_rows_before_after':[int(len(oldouter)),int(len(newouter))]},'layer_coordinate_fields':layer_comparisons,'app_source':{'before_path':f'git {args.before_revision}:apps/numilab_human_myosim_visual_probe.mm','before_sha256':digest(old_app),'after_path':'apps/numilab_human_myosim_visual_probe.mm','after_sha256':digest(new_app)},'mass_partition':{'before_formula_is_original_volume_times_density':source_debits_formula(old_app),'after_uses_explicit_debit_constants':('kWholeBodySkinMassDebitKg' in new_app and 'kWholeBodyFatMassDebitKg' in new_app),'debit_kg_before':old_debits,'debit_kg_after':new_debits,'per_tissue_exactly_unchanged':{n:old_debits[n]==new_debits[n] for n in debit_names},'total_debit_kg_before':sum(old_debits.values()),'total_debit_kg_after':sum(new_debits.values()),'total_debit_exactly_unchanged':sum(old_debits.values())==sum(new_debits.values()),'layer_volume_m3_before':old_vols,'layer_volume_m3_after':new_vols,'layer_volume_exactly_unchanged':{n:old_vols[n]==new_vols[n] for n in volume_names}},'source_payload_sha256_before':re.search(r'kWholeBodySkinSourcePayloadSHA256\[\]\s*=\s*"([0-9a-f]{64})"',old).group(1),'source_payload_sha256_after':re.search(r'kWholeBodySkinSourcePayloadSHA256\[\]\s*=\s*"([0-9a-f]{64})"',new).group(1),'audit_script_sha256':digest(Path(__file__).read_text())}
 out=args.output or root/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/mesh-quality-audit/source-preservation.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
